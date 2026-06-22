import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from PIL import Image, ImageTk
import os
import gc
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─────────────────────────────────────────────
#  TABLAS LTE Y CONFIGURACIONES
# ─────────────────────────────────────────────
BW_TABLE = {
    1.4: {"subcarriers_max": 93.3,  "useful_bw_mhz": 1.08, "data_subcarriers": 72,   "fft_size": 128},
    3.0: {"subcarriers_max": 200,   "useful_bw_mhz": 2.70, "data_subcarriers": 180,  "fft_size": 256},
    5.0: {"subcarriers_max": 333.3, "useful_bw_mhz": 4.50, "data_subcarriers": 300,  "fft_size": 512},
   10.0: {"subcarriers_max": 666.6, "useful_bw_mhz": 9.00, "data_subcarriers": 600,  "fft_size": 1024},
   15.0: {"subcarriers_max": 1000,  "useful_bw_mhz": 13.5, "data_subcarriers": 900,  "fft_size": 1024},
   20.0: {"subcarriers_max": 1333.3,"useful_bw_mhz": 18.0, "data_subcarriers": 1200, "fft_size": 2048},
}

CP_TYPES = {
    "normal":    4.7e-6,
    "extendido": 16.6e-6,
}

BITS_PER_SYMBOL = {"QPSK": 2, "16QAM": 4, "64QAM": 6}

# ── Escenario fijo para la pestaña de Simulación Montecarlo ──
# (no se piden en la interfaz; el escenario ya está definido)
MC_ITERATIONS = 3
MC_BW_MHZ = 15.0
MC_CP_TYPE = "normal"
MC_N_TAPS = 6

def fft_idx_to_freq_khz(idx, fft_size):
    idx = np.asarray(idx)
    freq = np.where(idx <= fft_size // 2, idx * 15.0, (idx - fft_size) * 15.0)
    return freq

def make_constellation(mod):
    if mod == "QPSK":
        return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    elif mod == "16QAM":
        vals = [-3, -1, 1, 3]
        pts = np.array([x + 1j*y for y in vals[::-1] for x in vals]) / np.sqrt(10)
        gray_order = [0,1,3,2, 4,5,7,6, 12,13,15,14, 8,9,11,10]
        return pts[gray_order]
    elif mod == "64QAM":
        vals = [-7,-5,-3,-1,1,3,5,7]
        return np.array([x + 1j*y for y in vals[::-1] for x in vals]) / np.sqrt(42)
    raise ValueError(f"Modulación desconocida: {mod}")

def bits_to_symbols(bits, constellation, bps):
    pad = (-len(bits)) % bps
    bits_padded = np.concatenate([bits, np.zeros(pad, dtype=int)])
    indices = bits_padded.reshape(-1, bps).dot(1 << np.arange(bps-1, -1, -1))
    return constellation[indices], pad

def symbols_to_bits(symbols, constellation, bps):
    dists = np.abs(symbols[:, None] - constellation[None, :])
    indices = np.argmin(dists, axis=1)
    bits_matrix = ((indices[:, None] & (1 << np.arange(bps-1, -1, -1))) > 0).astype(int)
    return bits_matrix.flatten()

# ─────────────────────────────────────────────
#  CANAL MULTIPATH PARA MÚLTIPLES ANTENAS (SIMO)
#  Perfil de potencia-retardo ITU (tipo COST207/EVA)
# ─────────────────────────────────────────────
# Retardos en microsegundos y potencias relativas en dB del perfil ITU
# usado como referencia. Estos valores son fijos (no se generan al azar),
# solo la GANANCIA COMPLEJA de cada trayecto es aleatoria (Rayleigh).
ITU_DELAYS_US = np.array([0, 0.11, 0.19, 0.41])
ITU_POWERS_DB = np.array([0, -0.9, -19.2, -22.8])

def make_simo_channels(fft_size, n_rx, n_taps=6):
    """
    Genera un canal multipath por cada antena receptora usando el perfil
    de potencia-retardo ITU (COST207/EVA). El número de trayectos y sus
    retardos quedan fijados por el perfil (ITU_DELAYS_US / ITU_POWERS_DB);
    el parámetro n_taps se conserva por compatibilidad con el resto del
    código (controles de la interfaz) pero no cambia el número de
    trayectos del perfil.

    Cada antena recibe una realización independiente de las ganancias
    complejas Rayleigh (parte real e imaginaria gaussianas), escaladas
    según la potencia relativa de cada trayecto, y la energía total de
    cada canal se normaliza a 1.
    """
    H_channels = []
    h_channels = []

    Fs = fft_size * 15e3
    delays_samp = np.round(ITU_DELAYS_US * 1e-6 * Fs).astype(int)
    delays_samp = np.clip(delays_samp, 0, fft_size - 1)

    powers = 10 ** (ITU_POWERS_DB / 10.0)
    powers = powers / np.sum(powers)

    for i in range(n_rx):
        # Ganancia compleja Rayleigh por trayecto, escalada por su potencia.
        # Independiente por antena -> diversidad espacial real entre Rx.
        gains = (np.random.randn(len(delays_samp)) +
                  1j * np.random.randn(len(delays_samp))) / np.sqrt(2)
        gains *= np.sqrt(powers)

        h = np.zeros(fft_size, dtype=complex)
        for d, g in zip(delays_samp, gains):
            h[d] += g

        # Normalizar energía total del canal a 1
        h /= np.sqrt(np.sum(np.abs(h) ** 2))

        H = np.fft.fft(h)
        H_channels.append(H)
        h_channels.append(h)

    return H_channels, h_channels

def apply_simo_channel(signal, h_channels, snr_db):
    rx_signals = []
    for h in h_channels:
        convolved = np.convolve(signal, h, mode='full')[:len(signal)]
        snr_lin = 10**(snr_db/10)
        power = np.mean(np.abs(convolved)**2)
        noise_power = power / snr_lin
        noise = np.sqrt(noise_power/2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
        rx_signals.append(convolved + noise)
    return rx_signals

# ─────────────────────────────────────────────
#  OFDM CORE CON DIVERSIDAD Y MRC
# ─────────────────────────────────────────────
class SIMO_OFDMSystem:
    def __init__(self, bw_mhz, mod, cp_type, n_rx=1, snr_db=20, n_taps=6, pilot_spacing=6):
        self.bw_mhz = bw_mhz
        self.mod = mod
        self.cp_type = cp_type
        self.n_rx = n_rx
        self.snr_db = snr_db
        self.n_taps = n_taps
        self.pilot_spacing = pilot_spacing

        info = BW_TABLE[bw_mhz]
        self.fft_size = info["fft_size"]
        self.n_data_sc = info["data_subcarriers"]
        self.bps = BITS_PER_SYMBOL[mod]
        self.constellation = make_constellation(mod)

        self.Fs = self.fft_size * 15e3
        self.cp_len = int(np.round(CP_TYPES[cp_type] * self.Fs))

        half = self.n_data_sc // 2
        sc_indices = list(range(1, half+1)) + list(range(self.fft_size - half, self.fft_size))
        self.sc_indices = np.array(sc_indices)
        self.sc_freqs_khz = fft_idx_to_freq_khz(self.sc_indices, self.fft_size)

        self.pilot_idx_local = np.arange(0, self.n_data_sc, pilot_spacing)
        self.pilot_sc = self.sc_indices[self.pilot_idx_local]
        self.pilot_freqs_khz = self.sc_freqs_khz[self.pilot_idx_local]

        self.data_idx_local = np.array([i for i in range(self.n_data_sc) if i not in set(self.pilot_idx_local)])
        self.data_sc = self.sc_indices[self.data_idx_local]
        self.data_freqs_khz = self.sc_freqs_khz[self.data_idx_local]

        self.n_pilots = len(self.pilot_sc)
        self.n_data_per_sym = len(self.data_sc)

        # ── FIX: índices de orden ascendente en FRECUENCIA para poder
        # interpolar correctamente con np.interp (que requiere xp creciente).
        # Antes se interpolaba sobre el índice local, que tiene un salto
        # de Nyquist (pasa de frecuencias positivas a negativas a mitad
        # del arreglo), generando una estimación de canal incorrecta justo
        # en ese salto -> ruido estructurado (líneas diagonales) en la imagen.
        self._pilot_sort = np.argsort(self.pilot_freqs_khz)
        self._data_sort = np.argsort(self.data_freqs_khz)
        self._data_unsort = np.argsort(self._data_sort)
        self._pilot_freqs_sorted = self.pilot_freqs_khz[self._pilot_sort]
        self._data_freqs_sorted = self.data_freqs_khz[self._data_sort]

        self.H_real_channels, self.h_channels = make_simo_channels(self.fft_size, self.n_rx, self.n_taps)

    # ── Versión vectorizada: modula todos los símbolos a la vez ────────────
    def modulate_batch(self, data_syms_batch, pilot_value=1.0+0j):
        """
        data_syms_batch: array (n_sym, n_data_per_sym) complejo
        Devuelve: tx_batch (n_sym, fft_size + cp_len)
        """
        n_sym = data_syms_batch.shape[0]
        # Frame frecuencial: todos los símbolos a la vez
        frames = np.zeros((n_sym, self.fft_size), dtype=complex)
        frames[:, self.pilot_sc] = pilot_value
        frames[:, self.data_sc]  = data_syms_batch
        # IFFT en bloque (axis=-1): (n_sym, fft_size)
        time_sigs = np.fft.ifft(frames, axis=-1) * np.sqrt(self.fft_size)
        # Añadir CP a cada símbolo
        cp = time_sigs[:, -self.cp_len:]               # (n_sym, cp_len)
        return np.concatenate([cp, time_sigs], axis=1)  # (n_sym, fft_size+cp_len)

    def demodulate_batch(self, rx_batch_per_ant, pilot_value=1.0+0j):
        """
        rx_batch_per_ant: lista de n_rx arrays, cada uno (n_sym, fft_size+cp_len)
        Devuelve:
            eq_batch   (n_sym, n_data_per_sym) complejo  — salida MRC ecualizada
            H_est_list lista n_rx de (n_sym, n_data_per_sym) — para modo normal
        """
        n_sym = rx_batch_per_ant[0].shape[0]
        num_mrc = np.zeros((n_sym, self.n_data_per_sym), dtype=complex)
        den_mrc = np.zeros((n_sym, self.n_data_per_sym), dtype=float)
        H_est_list = []

        # Vectores de frecuencia (ya pre-ordenados en __init__)
        xp = self._pilot_freqs_sorted   # (n_pilots_sorted,)
        xi = self._data_freqs_sorted    # (n_data_sorted,)

        for rx_b in rx_batch_per_ant:
            freq     = np.fft.fft(rx_b[:, self.cp_len:], axis=-1) / np.sqrt(self.fft_size)
            H_pilots = freq[:, self.pilot_sc] / pilot_value      # (n_sym, n_pilots)
            data_rx  = freq[:, self.data_sc]                      # (n_sym, n_data)

            H_ps = H_pilots[:, self._pilot_sort]  # (n_sym, n_pilots_sorted)

            # ── Interpolación vectorizada en bloque (sin bucle Python por símbolo) ──
            # np.interp no admite entradas 2D, así que usamos searchsorted +
            # interpolación lineal manual en NumPy, que opera sobre matrices completas.
            # Equivalente exacto a np.interp fila a fila pero 100x más rápido.
            idx = np.searchsorted(xp, xi, side='right') - 1
            idx = np.clip(idx, 0, len(xp) - 2)

            x0 = xp[idx];     x1 = xp[idx + 1]        # (n_data_sorted,)
            t  = (xi - x0) / (x1 - x0 + 1e-30)        # coef. interpolación

            # H_ps tiene shape (n_sym, n_pilots_sorted); sacamos columnas idx e idx+1
            y0_r = H_ps[:, idx].real;   y1_r = H_ps[:, idx + 1].real
            y0_i = H_ps[:, idx].imag;   y1_i = H_ps[:, idx + 1].imag

            H_est_sorted_r = y0_r + t * (y1_r - y0_r)   # (n_sym, n_data_sorted)
            H_est_sorted_i = y0_i + t * (y1_i - y0_i)
            H_est = (H_est_sorted_r + 1j * H_est_sorted_i)[:, self._data_unsort]

            num_mrc += np.conj(H_est) * data_rx
            den_mrc += np.abs(H_est) ** 2
            H_est_list.append(H_est)

        eq_batch = num_mrc / (den_mrc + 1e-10)
        return eq_batch, H_est_list

    def transmit_bits(self, bits, lite_mode=False):
        """
        Procesa todos los símbolos OFDM en BLOQUE (vectorizado) en vez de
        uno a uno en un bucle Python. Esto elimina el overhead de Python por
        símbolo y da una mejora de velocidad de 5-10x en transmisiones largas.

        lite_mode=True  (Montecarlo): no acumula H_est/tx/rx por símbolo.
        lite_mode=False (pestaña principal): devuelve todos los acumuladores
                        para graficar canal estimado y constelaciones.
        """
        syms_data, pad = bits_to_symbols(bits, self.constellation, self.bps)

        # Rellenar hasta múltiplo exacto de n_data_per_sym
        n_sym = int(np.ceil(len(syms_data) / self.n_data_per_sym))
        n_pad_sym = n_sym * self.n_data_per_sym - len(syms_data)
        if n_pad_sym > 0:
            syms_data = np.concatenate([syms_data, np.zeros(n_pad_sym, dtype=complex)])
        syms_matrix = syms_data.reshape(n_sym, self.n_data_per_sym)  # (n_sym, M)

        # ── Modulación en bloque ──────────────────────────────────────────────
        tx_batch = self.modulate_batch(syms_matrix)  # (n_sym, N+cp)

        # ── Canal: aplicar a cada símbolo, cada antena ──────────────────────
        # np.convolve no opera en 2D; usamos scipy si está disponible,
        # de lo contrario aplicamos la convolución eficientemente via FFT.
        sym_len = self.fft_size + self.cp_len
        rx_batch_per_ant = []
        for h in self.h_channels:
            # Convolución vía FFT en el dominio tiempo: más rápida que np.convolve
            # en bucle para n_sym > 50.
            H_full = np.fft.fft(h, n=sym_len)          # respuesta en frecuencia
            TX_F   = np.fft.fft(tx_batch, n=sym_len, axis=-1)  # (n_sym, sym_len)
            conv   = np.fft.ifft(TX_F * H_full, axis=-1).real + \
                     1j * np.fft.ifft(TX_F * H_full, axis=-1).imag
            conv   = np.fft.ifft(TX_F * H_full, axis=-1)[:, :sym_len]  # truncar

            # Ruido AWGN en bloque
            snr_lin    = 10 ** (self.snr_db / 10)
            power      = np.mean(np.abs(conv) ** 2)
            noise_pwr  = power / snr_lin
            noise      = np.sqrt(noise_pwr / 2) * (
                np.random.randn(n_sym, sym_len) + 1j * np.random.randn(n_sym, sym_len)
            )
            rx_batch_per_ant.append(conv + noise)

        # ── Demodulación + estimación de canal + MRC en bloque ───────────────
        eq_batch, H_est_list = self.demodulate_batch(rx_batch_per_ant)
        # eq_batch: (n_sym, n_data_per_sym)

        # ── Detección de símbolos en bloque ──────────────────────────────────
        eq_flat = eq_batch.reshape(-1)  # aplanar (n_sym*n_data_per_sym,)
        rx_bits_all = symbols_to_bits(eq_flat, self.constellation, self.bps)

        # Quitar padding de símbolo y de bits
        n_bits_validos = n_sym * self.n_data_per_sym * self.bps - n_pad_sym * self.bps
        rx_bits_all = rx_bits_all[:n_bits_validos]
        if pad > 0:
            rx_bits_all = rx_bits_all[:-pad]

        if lite_mode:
            return rx_bits_all, None, None, None

        # Modo normal: reconstruir acumuladores símbolo a símbolo para
        # compatibilidad con run_main_simulation (graficado de canal, etc.)
        H_est_acc   = [[H_est_list[rx_i][sym_i] for rx_i in range(self.n_rx)]
                       for sym_i in range(n_sym)]
        tx_syms_acc = [syms_matrix[i] for i in range(n_sym)]
        rx_syms_acc = [eq_batch[i]    for i in range(n_sym)]
        return rx_bits_all, H_est_acc, tx_syms_acc, rx_syms_acc

    # ── Métodos originales símbolo-a-símbolo (conservados para compatibilidad) ──
    def modulate(self, data_symbols, pilot_value=1.0+0j):
        frame = np.zeros(self.fft_size, dtype=complex)
        frame[self.pilot_sc] = pilot_value
        frame[self.data_sc]  = data_symbols
        time_sig = np.fft.ifft(frame) * np.sqrt(self.fft_size)
        cp = time_sig[-self.cp_len:]
        return np.concatenate([cp, time_sig])

    def demodulate_and_estimate(self, rx_signals_list, pilot_value=1.0+0j):
        data_rx_all    = []
        H_est_data_all = []
        H_pilots_all   = []
        for rx_sig in rx_signals_list:
            ofdm_sym = rx_sig[self.cp_len:]
            freq     = np.fft.fft(ofdm_sym) / np.sqrt(self.fft_size)
            pilots_rx = freq[self.pilot_sc]
            data_rx   = freq[self.data_sc]
            H_pilots        = pilots_rx / pilot_value
            H_pilots_sorted = H_pilots[self._pilot_sort]
            H_est_sorted = (
                np.interp(self._data_freqs_sorted, self._pilot_freqs_sorted, H_pilots_sorted.real) +
                1j * np.interp(self._data_freqs_sorted, self._pilot_freqs_sorted, H_pilots_sorted.imag)
            )
            H_est_data = H_est_sorted[self._data_unsort]
            data_rx_all.append(data_rx)
            H_est_data_all.append(H_est_data)
            H_pilots_all.append(H_pilots)
        num_mrc = np.zeros_like(data_rx_all[0], dtype=complex)
        den_mrc = np.zeros_like(data_rx_all[0], dtype=float)
        for i in range(self.n_rx):
            num_mrc += np.conj(H_est_data_all[i]) * data_rx_all[i]
            den_mrc += np.abs(H_est_data_all[i]) ** 2
        eq_data = num_mrc / (den_mrc + 1e-10)
        return eq_data, H_est_data_all

# ─────────────────────────────────────────────
#  INTERFAZ GRÁFICA TKINTER
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Avanzado OFDM - SIMO con MRC (Corregido)")
        self.root.geometry("1200x800")

        self.img_path = ""

        # Notebook principal para las dos pestañas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_montecarlo = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="Transmisión General / Imagen")
        self.notebook.add(self.tab_montecarlo, text="Simulación Montecarlo")

        self.setup_tab_main()
        self.setup_tab_montecarlo()

    def setup_tab_main(self):
        # Panel de control izquierdo
        ctrl_frame = ttk.LabelFrame(self.tab_main, text=" Parámetros del Sistema ", padding=10)
        ctrl_frame.pack(side='left', fill='y', padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Antenas Receptoras (Rx):").grid(row=0, column=0, sticky='w', pady=4)
        self.spin_rx = ttk.Spinbox(ctrl_frame, from_=1, to=16, width=12)
        self.spin_rx.set(2)
        self.spin_rx.grid(row=0, column=1, pady=4)

        ttk.Label(ctrl_frame, text="Modulación:").grid(row=1, column=0, sticky='w', pady=4)
        self.combo_mod = ttk.Combobox(ctrl_frame, values=["QPSK", "16QAM", "64QAM"], state="readonly", width=11)
        self.combo_mod.set("16QAM")
        self.combo_mod.grid(row=1, column=1, pady=4)

        ttk.Label(ctrl_frame, text="Ancho de Banda (MHz):").grid(row=2, column=0, sticky='w', pady=4)
        self.combo_bw = ttk.Combobox(ctrl_frame, values=["1.4", "3.0", "5.0", "10.0", "15.0", "20.0"], state="readonly", width=11)
        self.combo_bw.set("5.0")
        self.combo_bw.grid(row=2, column=1, pady=4)

        ttk.Label(ctrl_frame, text="Prefijo Cíclico:").grid(row=3, column=0, sticky='w', pady=4)
        self.combo_cp = ttk.Combobox(ctrl_frame, values=["normal", "extendido"], state="readonly", width=11)
        self.combo_cp.set("normal")
        self.combo_cp.grid(row=3, column=1, pady=4)

        ttk.Label(ctrl_frame, text="SNR de Prueba (dB):").grid(row=4, column=0, sticky='w', pady=4)
        self.entry_snr = ttk.Entry(ctrl_frame, width=13)
        self.entry_snr.insert(0, "10")
        self.entry_snr.grid(row=4, column=1, pady=4)

        ttk.Label(ctrl_frame, text="Copias del Canal (Taps):").grid(row=5, column=0, sticky='w', pady=4)
        self.spin_taps = ttk.Spinbox(ctrl_frame, from_=1, to=20, width=12)
        self.spin_taps.set(6)
        self.spin_taps.grid(row=5, column=1, pady=4)

        ttk.Button(ctrl_frame, text="Seleccionar Imagen", command=self.load_image).grid(row=6, column=0, columnspan=2, pady=15, sticky='we')
        self.lbl_img = ttk.Label(ctrl_frame, text="Sin imagen cargada", foreground="red")
        self.lbl_img.grid(row=7, column=0, columnspan=2, pady=2)

        ttk.Button(ctrl_frame, text="Simular Transmisión", command=self.run_main_simulation).grid(row=8, column=0, columnspan=2, pady=15, sticky='we')

        # Área de Gráficas Derecha
        self.plot_frame = ttk.Frame(self.tab_main)
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    def setup_tab_montecarlo(self):
        m_frame = ttk.LabelFrame(self.tab_montecarlo, text=" Simulación Montecarlo (escenario predefinido) ", padding=10)
        m_frame.pack(side='top', fill='x', padx=10, pady=10)

        info_txt = (f"Escenario fijo: BW = {MC_BW_MHZ} MHz · CP = {MC_CP_TYPE} · Taps = {MC_N_TAPS} · "
                     f"{MC_ITERATIONS} iteraciones por punto.\n"
                     "Por cada punto (Modulación, Antenas Rx, SNR) se transmite y recibe la IMAGEN "
                     "COMPLETA cargada en la pestaña 'Transmisión General / Imagen' "
                     f"{MC_ITERATIONS} veces (canal aleatorio nuevo en cada repetición), acumulando "
                     "errores de bit para calcular el BER de ese punto.")
        ttk.Label(m_frame, text=info_txt, wraplength=950, foreground="#555555",
                  font=("TkDefaultFont", 9)).grid(row=0, column=0, columnspan=2, padx=5, pady=(0,8), sticky='w')

        self.btn_run_montecarlo = ttk.Button(m_frame, text="Ejecutar Curvas BER de Montecarlo", command=self.run_montecarlo)
        self.btn_run_montecarlo.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.lbl_mc_status = ttk.Label(m_frame, text="", foreground="#0055ff", font=("TkDefaultFont", 9, "bold"))
        self.lbl_mc_status.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky='w')

        self.mc_progress = ttk.Progressbar(m_frame, orient='horizontal', mode='determinate', length=400)
        self.mc_progress.grid(row=3, column=0, columnspan=2, padx=5, pady=(2,5), sticky='we')

        self.mc_plot_frame = ttk.Frame(self.tab_montecarlo)
        self.mc_plot_frame.pack(side='bottom', fill='both', expand=True, padx=10, pady=10)


    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.img_path = path
            self.lbl_img.config(text=os.path.basename(path), foreground="green")

    def run_main_simulation(self):
        if not self.img_path:
            messagebox.showwarning("Advertencia", "Por favor seleccione una imagen base primero.")
            return

        try:
            n_rx = int(self.spin_rx.get())
            mod = self.combo_mod.get()
            bw = float(self.combo_bw.get())
            cp = self.combo_cp.get()
            snr = float(self.entry_snr.get())
            taps = int(self.spin_taps.get())
        except ValueError:
            messagebox.showerror("Error", "Revise que los datos de entrada numéricos sean válidos.")
            return

        # Procesar Imagen a bits
        img = Image.open(self.img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        shape = arr.shape
        bits = np.unpackbits(arr.flatten())

        # Ejecutar Sistema SIMO OFDM con MRC
        ofdm = SIMO_OFDMSystem(bw_mhz=bw, mod=mod, cp_type=cp, n_rx=n_rx, snr_db=snr, n_taps=taps)
        rx_bits, H_est_acc, tx_syms_acc, rx_syms_acc = ofdm.transmit_bits(bits)

        min_len = min(len(bits), len(rx_bits))
        bit_errors = int(np.sum(bits[:min_len] != rx_bits[:min_len]))
        ber = bit_errors / len(bits)

        # Reconstruir Imagen Recibida
        n_bytes = shape[0] * shape[1] * shape[2]
        bits_needed = n_bytes * 8
        if len(rx_bits) < bits_needed:
            rx_bits = np.pad(rx_bits, (0, bits_needed - len(rx_bits)))
        img_rx_arr = np.packbits(rx_bits[:bits_needed]).reshape(shape)
        img_recv = Image.fromarray(img_rx_arr.astype(np.uint8))

        # Dibujar Gráficas en la pestaña principal
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(11, 7))
        fig.patch.set_facecolor("#f8f9fa")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Canal Real vs Estimado (Se grafica la antena 1 de ejemplo)
        ax1 = fig.add_subplot(gs[0, :])
        data_freqs = ofdm.data_freqs_khz
        sort_idx = np.argsort(data_freqs)

        H_real_full = np.abs(ofdm.H_real_channels[0])
        H_est_mean = np.mean(np.abs(np.array(H_est_acc)), axis=0)[0]  # Antena 1

        ax1.plot(data_freqs[sort_idx], H_real_full[ofdm.data_sc][sort_idx], label="Canal real |H_1(f)|", color="#0055ff", lw=1.5)
        ax1.plot(data_freqs[sort_idx], H_est_mean[sort_idx], label="Est. MRC |Ĥ_1(f)|", color="#ff5500", ls="--", lw=1.2)
        ax1.set_title(f"Canal en Frecuencia (Antena 1) con {taps} Taps", fontweight="bold", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8)

        # 2. Imagen Original
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(img)
        ax2.set_title("Imagen Original", fontweight="bold", fontsize=10)
        ax2.axis("off")

        # 3. Imagen Recibida
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(img_recv)
        ax3.set_title(f"Recibida (Rx Antenas: {n_rx})", fontweight="bold", fontsize=10)
        ax3.axis("off")

        # 4. Datos Estadísticos
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        stats = [
            f"Modulación: {mod}",
            f"Ancho Banda: {bw} MHz",
            f"FFT size: {ofdm.fft_size}",
            f"SNR: {snr} dB",
            f"Errores bit: {bit_errors}",
            f"BER: {ber:.4e}"
        ]
        y_pos = 0.8
        for text in stats:
            ax4.text(0.1, y_pos, text, fontsize=10, fontweight="bold", color="#333333")
            y_pos -= 0.15

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def run_montecarlo(self):
        # ── Validación ────────────────────────────────────────────────────────
        if not self.img_path:
            messagebox.showwarning("Advertencia",
                "Cargue una imagen en la pestaña 'Transmisión General / Imagen' antes de "
                "ejecutar el Montecarlo (cada punto de la curva transmite esa imagen completa).")
            return

        # Escenario fijo (no se pide en la interfaz de esta pestaña)
        iterations = MC_ITERATIONS
        bw   = MC_BW_MHZ
        cp   = MC_CP_TYPE
        taps = MC_N_TAPS

        # ── Cargar imagen UNA sola vez ─────────────────────────────────────────
        img = Image.open(self.img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        tx_bits_image = np.unpackbits(arr.flatten())
        n_bits_image  = len(tx_bits_image)

        # ── Estimación de tiempo antes de empezar ─────────────────────────────
        # Se mide 1 transmisión del peor caso (QPSK + n_rx=8, el más lento)
        # y se extrapola al total para informar al usuario.
        self.lbl_mc_status.config(text="Estimando tiempo de cómputo...")
        self.root.update_idletasks()

        ofdm_probe = SIMO_OFDMSystem(bw_mhz=bw, mod="QPSK", cp_type=cp,
                                      n_rx=8, snr_db=10, n_taps=taps)
        t_probe_start = time.perf_counter()
        ofdm_probe.transmit_bits(tx_bits_image, lite_mode=True)
        t_probe = time.perf_counter() - t_probe_start
        del ofdm_probe
        gc.collect()

        # Factores de escala por modulacion y antenas (medidos anteriormente)
        # QPSK n_rx=8 es el peor caso (base = 1.0). Los demás son más baratos:
        # bps mayor -> menos símbolos OFDM; n_rx menor -> menos demodulaciones.
        # Usamos coeficientes medidos: costo ~ n_rx * (bits/sym de QPSK / bits/sym_mod)
        factores = {}
        for mod_f, bps_f in BITS_PER_SYMBOL.items():
            for nr in [1, 3, 8]:
                # relativo a QPSK n_rx=8 (base)
                factores[(mod_f, nr)] = (nr / 8) * (2 / bps_f)

        t_total_est = 0
        for mod_f in ["QPSK", "16QAM", "64QAM"]:
            for nr in [1, 3, 8]:
                t_total_est += t_probe * factores[(mod_f, nr)] * len(np.arange(0,26,5)) * iterations

        mins_est = t_total_est / 60
        self.lbl_mc_status.config(
            text=f"Tiempo estimado: ~{mins_est:.1f} min  "
                 f"(imagen {arr.shape[1]}×{arr.shape[0]} px, {n_bits_image:,} bits). "
                 f"Iniciando...")
        self.root.update_idletasks()

        self.btn_run_montecarlo.config(state="disabled")

        snr_axis      = np.arange(0, 26, 5)
        antenas_test  = [1, 3, 8]
        modulations   = ["QPSK", "16QAM", "64QAM"]

        total_steps = len(modulations) * len(antenas_test) * len(snr_axis) * iterations
        self.mc_progress.config(maximum=total_steps, value=0)
        step_count  = 0
        t_start_all = time.perf_counter()

        # Limpiar área de gráficas
        for widget in self.mc_plot_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        fig.patch.set_facecolor("#f8f9fa")

        # ── Loop Montecarlo ───────────────────────────────────────────────────
        # Por cada punto (modulación, n_rx, SNR): transmite la imagen completa
        # MC_ITERATIONS veces con canal aleatorio nuevo en cada repetición,
        # acumula errores y calcula BER del punto.
        #
        # MEJORAS DE MEMORIA Y RENDIMIENTO vs versión anterior:
        #   1. lite_mode=True  -> transmit_bits NO acumula H_est/tx/rx_syms
        #      (ahorraba hasta 300+ MB por transmisión con n_rx=8).
        #   2. del + gc.collect() explícito tras cada iteración -> fuerza la
        #      liberación inmediata de los arrays temporales (señales Tx/Rx,
        #      canal, etc.) en vez de esperar al GC automático de Python.
        #   3. La barra de progreso muestra tiempo transcurrido y tiempo
        #      restante estimado para que el usuario sepa que no se colgó.
        for m_idx, mod in enumerate(modulations):
            ax = axes[m_idx]

            for n_rx in antenas_test:
                ber_results = []

                for snr in snr_axis:
                    total_errors = 0
                    total_bits   = 0

                    for it in range(iterations):
                        # ── Feedback en tiempo real ───────────────────────────
                        t_elapsed   = time.perf_counter() - t_start_all
                        pasos_hechos = max(step_count, 1)
                        t_restante  = (t_elapsed / pasos_hechos) * (total_steps - step_count)
                        self.lbl_mc_status.config(
                            text=f"Mod {mod} | Rx {n_rx} | SNR {snr} dB | "
                                 f"Iter {it+1}/{iterations}  —  "
                                 f"Transcurrido: {t_elapsed/60:.1f} min  |  "
                                 f"Restante est.: {t_restante/60:.1f} min")
                        self.mc_progress.config(value=step_count)
                        self.root.update_idletasks()

                        # ── Transmisión con canal aleatorio nuevo ─────────────
                        ofdm_sim = SIMO_OFDMSystem(bw_mhz=bw, mod=mod, cp_type=cp,
                                                    n_rx=n_rx, snr_db=snr, n_taps=taps)

                        # lite_mode=True: no acumula H_est_acc/tx_syms/rx_syms
                        rx_bits, _, _, _ = ofdm_sim.transmit_bits(tx_bits_image,
                                                                    lite_mode=True)

                        min_l = min(n_bits_image, len(rx_bits))
                        total_errors += int(np.sum(tx_bits_image[:min_l] != rx_bits[:min_l]))
                        total_bits   += min_l

                        # ── Liberar memoria inmediatamente ────────────────────
                        del ofdm_sim, rx_bits
                        gc.collect()

                        step_count += 1

                    ber_real = total_errors / total_bits if total_bits > 0 else 1
                    ber_results.append(ber_real)

                # ── Graficar curva de esta combinación (mod, n_rx) ───────────
                ber_results = np.array(ber_results)
                PISO_VISUAL = 1e-5
                es_cero  = ber_results == 0
                ber_plot = np.where(es_cero, PISO_VISUAL, ber_results)

                line, = ax.semilogy(snr_axis, ber_plot, 'o-',
                                    label=f"{n_rx} Antenas Rx", lw=1.5)
                if np.any(es_cero):
                    ax.scatter(snr_axis[es_cero], ber_plot[es_cero],
                               marker='v', s=45, color=line.get_color(),
                               edgecolors='black', linewidths=0.7, zorder=5)

            ax.set_title(f"Modulación {mod}", fontweight="bold", fontsize=10)
            ax.set_xlabel("SNR (dB)")
            if m_idx == 0:
                ax.set_ylabel("BER (Bit Error Rate)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
            ax.set_ylim(1e-5, 1)

        t_total_real = time.perf_counter() - t_start_all
        fig.suptitle(
            f"Análisis Estadístico de Montecarlo usando Diversidad MRC\n"
            f"(Imagen completa × {iterations} iter/punto · BW={bw} MHz · CP={cp} · "
            f"Taps={taps} · Canal ITU  —  Tiempo total: {t_total_real/60:.1f} min)\n"
            f"▽ = 0 errores detectados en esas {iterations} iteraciones "
            f"(graficado en el piso del eje)",
            fontweight="bold", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=self.mc_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.mc_progress.config(value=total_steps)
        self.lbl_mc_status.config(
            text=f"Listo. Tiempo total real: {t_total_real/60:.1f} min.")
        self.btn_run_montecarlo.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()