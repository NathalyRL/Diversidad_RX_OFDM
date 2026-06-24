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

MC_ITERATIONS = 3
MC_BW_MHZ = 20.0
MC_CP_TYPE = "normal"

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
# ─────────────────────────────────────────────
ITU_DELAYS_US = np.array([0, 0.11, 0.19, 0.41])
ITU_POWERS_DB = np.array([0, -0.9, -19.2, -22.8])

def make_simo_channels(fft_size, n_rx):
    H_channels = []
    h_channels = []

    Fs = fft_size * 15e3
    delays_samp = np.round(ITU_DELAYS_US * 1e-6 * Fs).astype(int)
    delays_samp = np.clip(delays_samp, 0, fft_size - 1)

    powers = 10 ** (ITU_POWERS_DB / 10.0)
    powers = powers / np.sum(powers)

    for i in range(n_rx):
        gains = (np.random.randn(len(delays_samp)) + 1j * np.random.randn(len(delays_samp))) / np.sqrt(2)
        gains *= np.sqrt(powers)

        h = np.zeros(fft_size, dtype=complex)
        for d, g in zip(delays_samp, gains):
            h[d] += g

        h /= np.sqrt(np.sum(np.abs(h) ** 2))

        H = np.fft.fft(h)
        H_channels.append(H)
        h_channels.append(h)

    return H_channels, h_channels

# ─────────────────────────────────────────────
#  OFDM CORE CON DIVERSIDAD Y MRC
# ─────────────────────────────────────────────
class SIMO_OFDMSystem:
    def __init__(self, bw_mhz, mod, cp_type, n_rx=1, snr_db=20, pilot_spacing=6):
        self.bw_mhz = bw_mhz
        self.mod = mod
        self.cp_type = cp_type
        self.n_rx = n_rx
        self.snr_db = snr_db
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

        self._pilot_sort = np.argsort(self.pilot_freqs_khz)
        self._data_sort = np.argsort(self.data_freqs_khz)
        self._data_unsort = np.argsort(self._data_sort)
        self._pilot_freqs_sorted = self.pilot_freqs_khz[self._pilot_sort]
        self._data_freqs_sorted = self.data_freqs_khz[self._data_sort]

        self.H_real_channels, self.h_channels = make_simo_channels(self.fft_size, self.n_rx)

    def modulate_batch(self, data_syms_batch, pilot_value=1.0+0j):
        n_sym = data_syms_batch.shape[0]
        frames = np.zeros((n_sym, self.fft_size), dtype=complex)
        frames[:, self.pilot_sc] = pilot_value
        frames[:, self.data_sc]  = data_syms_batch
        time_sigs = np.fft.ifft(frames, axis=-1) * np.sqrt(self.fft_size)
        cp = time_sigs[:, -self.cp_len:]
        return np.concatenate([cp, time_sigs], axis=1)

    def demodulate_batch(self, rx_batch_per_ant, pilot_value=1.0+0j):
        n_sym = rx_batch_per_ant[0].shape[0]
        num_mrc = np.zeros((n_sym, self.n_data_per_sym), dtype=complex)
        den_mrc = np.zeros((n_sym, self.n_data_per_sym), dtype=float)
        H_est_list = []

        xp = self._pilot_freqs_sorted
        xi = self._data_freqs_sorted

        for rx_b in rx_batch_per_ant:
            freq     = np.fft.fft(rx_b[:, self.cp_len:], axis=-1) / np.sqrt(self.fft_size)
            H_pilots = freq[:, self.pilot_sc] / pilot_value
            data_rx  = freq[:, self.data_sc]

            H_ps = H_pilots[:, self._pilot_sort]

            idx = np.searchsorted(xp, xi, side='right') - 1
            idx = np.clip(idx, 0, len(xp) - 2)

            x0 = xp[idx];     x1 = xp[idx + 1]
            t  = (xi - x0) / (x1 - x0 + 1e-30)

            y0_r = H_ps[:, idx].real;   y1_r = H_ps[:, idx + 1].real
            y0_i = H_ps[:, idx].imag;   y1_i = H_ps[:, idx + 1].imag

            H_est_sorted_r = y0_r + t * (y1_r - y0_r)
            H_est_sorted_i = y0_i + t * (y1_i - y0_i)
            H_est = (H_est_sorted_r + 1j * H_est_sorted_i)[:, self._data_unsort]

            num_mrc += np.conj(H_est) * data_rx
            den_mrc += np.abs(H_est) ** 2
            H_est_list.append(H_est)

        eq_batch = num_mrc / (den_mrc + 1e-10)
        return eq_batch, H_est_list

    def transmit_bits(self, bits, lite_mode=False):
        syms_data, pad = bits_to_symbols(bits, self.constellation, self.bps)

        n_sym = int(np.ceil(len(syms_data) / self.n_data_per_sym))
        n_pad_sym = n_sym * self.n_data_per_sym - len(syms_data)
        if n_pad_sym > 0:
            syms_data = np.concatenate([syms_data, np.zeros(n_pad_sym, dtype=complex)])
        syms_matrix = syms_data.reshape(n_sym, self.n_data_per_sym)

        tx_batch = self.modulate_batch(syms_matrix)

        sym_len = self.fft_size + self.cp_len
        rx_batch_per_ant = []
        for h in self.h_channels:
            H_full = np.fft.fft(h, n=sym_len)
            TX_F   = np.fft.fft(tx_batch, n=sym_len, axis=-1)
            conv   = np.fft.ifft(TX_F * H_full, axis=-1)[:, :sym_len]

            snr_lin    = 10 ** (self.snr_db / 10)
            power      = np.mean(np.abs(conv) ** 2)
            noise_pwr  = power / snr_lin
            noise      = np.sqrt(noise_pwr / 2) * (
                np.random.randn(n_sym, sym_len) + 1j * np.random.randn(n_sym, sym_len)
            )
            rx_batch_per_ant.append(conv + noise)

        eq_batch, H_est_list = self.demodulate_batch(rx_batch_per_ant)

        eq_flat = eq_batch.reshape(-1)
        rx_bits_all = symbols_to_bits(eq_flat, self.constellation, self.bps)

        n_bits_validos = n_sym * self.n_data_per_sym * self.bps - n_pad_sym * self.bps
        rx_bits_all = rx_bits_all[:n_bits_validos]
        if pad > 0:
            rx_bits_all = rx_bits_all[:-pad]

        if lite_mode:
            return rx_bits_all, None, None, None

        H_est_acc   = [[H_est_list[rx_i][sym_i] for rx_i in range(self.n_rx)] for sym_i in range(n_sym)]
        tx_syms_acc = [syms_matrix[i] for i in range(n_sym)]
        rx_syms_acc = [eq_batch[i] for i in range(n_sym)]
        return rx_bits_all, H_est_acc, tx_syms_acc, rx_syms_acc

# ─────────────────────────────────────────────
#  INTERFAZ GRÁFICA TKINTER
# ─────────────────────────────────────────────
class App:
    def on_closing(self):
        # Cambiamos la bandera para detener los bucles activos
        self.running = False
        # Destruimos la ventana de Tkinter
        self.root.destroy()
        # Forzamos la salida del proceso de Python inmediatamente
        import sys
        sys.exit(0)

    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM - SIMO con MRC")
        self.root.geometry("1240x820")

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.img_path = ""
        self.mc_history = {"QPSK": {}, "16QAM": {}, "64QAM": {}}
        
        # FIX: sharey=False para permitir que cada eje Y se auto-ajuste de forma independiente
        self.fig_mc, self.axes_mc = plt.subplots(1, 3, figsize=(12, 4.8), sharey=False)
        self.fig_mc.patch.set_facecolor("#f8f9fa")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_montecarlo = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="Transmisión General / Imagen")
        self.notebook.add(self.tab_montecarlo, text="Simulación Montecarlo")

        self.setup_tab_main()
        self.setup_tab_montecarlo()

    def setup_tab_main(self):
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
        self.entry_snr.insert(0, "10.0")
        self.entry_snr.grid(row=4, column=1, pady=4)

        # ttk.Label(ctrl_frame, text="Copias del Canal (Taps):").grid(row=5, column=0, sticky='w', pady=4)
        # self.spin_taps = ttk.Spinbox(ctrl_frame, from_=1, to=20, width=12)
        # self.spin_taps.set(6)
        # self.spin_taps.grid(row=5, column=1, pady=4)

        ttk.Button(ctrl_frame, text="Seleccionar Imagen", command=self.load_image).grid(row=6, column=0, columnspan=2, pady=15, sticky='we')
        self.lbl_img = ttk.Label(ctrl_frame, text="Sin imagen cargada", foreground="red")
        self.lbl_img.grid(row=7, column=0, columnspan=2, pady=2)

        ttk.Button(ctrl_frame, text="Simular Transmisión", command=self.run_main_simulation).grid(row=8, column=0, columnspan=2, pady=15, sticky='we')

        self.plot_frame = ttk.Frame(self.tab_main)
        self.plot_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    def setup_tab_montecarlo(self):
        m_frame = ttk.LabelFrame(self.tab_montecarlo, text=" Ejecución Modular por Esquema ", padding=10)
        m_frame.pack(side='top', fill='x', padx=10, pady=10)

        info_txt = (f"Escenario fijo: BW = {MC_BW_MHZ} MHz · CP = {MC_CP_TYPE}  "
                    f"{MC_ITERATIONS} iteraciones por punto.\n"
                    "Cada modulación ajustará automáticamente los límites de su eje Y para visualizar adecuadamente caídas bajas de BER.")
        ttk.Label(m_frame, text=info_txt, wraplength=1100, foreground="#555555",
                  font=("TkDefaultFont", 9)).grid(row=0, column=0, columnspan=3, padx=5, pady=(0,8), sticky='w')

        self.btn_mc_qpsk = ttk.Button(m_frame, text="Ejecutar QPSK", command=lambda: self.run_select_montecarlo("QPSK"))
        self.btn_mc_qpsk.grid(row=1, column=0, padx=5, pady=5, sticky='we')

        self.btn_mc_16qam = ttk.Button(m_frame, text="Ejecutar 16-QAM", command=lambda: self.run_select_montecarlo("16QAM"))
        self.btn_mc_16qam.grid(row=1, column=1, padx=5, pady=5, sticky='we')

        self.btn_mc_64qam = ttk.Button(m_frame, text="Ejecutar 64-QAM", command=lambda: self.run_select_montecarlo("64QAM"))
        self.btn_mc_64qam.grid(row=1, column=2, padx=5, pady=5, sticky='we')

        self.lbl_mc_status = ttk.Label(m_frame, text="Esperando ejecución...", foreground="#0055ff", font=("TkDefaultFont", 9, "bold"))
        self.lbl_mc_status.grid(row=2, column=0, columnspan=3, padx=5, pady=2, sticky='w')

        self.mc_progress = ttk.Progressbar(m_frame, orient='horizontal', mode='determinate', length=400)
        self.mc_progress.grid(row=3, column=0, columnspan=3, padx=5, pady=(2,5), sticky='we')

        self.canvas_mc_frame = ttk.Frame(self.tab_montecarlo)
        self.canvas_mc_frame.pack(side='bottom', fill='both', expand=True, padx=10, pady=10)
        
        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, master=self.canvas_mc_frame)
        self.canvas_mc.get_tk_widget().pack(fill='both', expand=True)
        self.render_empty_mc_plots()

    def render_empty_mc_plots(self):
        modulations = ["QPSK", "16QAM", "64QAM"]
        for idx, ax in enumerate(self.axes_mc):
            ax.clear()
            ax.set_facecolor("#fbfcfe")
            ax.set_title(f"Modulación {modulations[idx]}", fontweight="bold", fontsize=10)
            ax.set_xlabel("SNR (dB)", fontsize=8)
            ax.set_ylabel("BER (Bit Error Rate)", fontsize=8)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1.2)  # Límites base por defecto
            ax.set_xlim(0, 25)
        self.fig_mc.suptitle("Análisis Montecarlo", fontweight="bold", fontsize=12)
        self.canvas_mc.draw()

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
            
        except ValueError:
            messagebox.showerror("Error", "Revise que los datos de entrada numéricos sean válidos.")
            return

        img = Image.open(self.img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        shape = arr.shape
        bits = np.unpackbits(arr.flatten())

        ofdm = SIMO_OFDMSystem(bw_mhz=bw, mod=mod, cp_type=cp, n_rx=n_rx, snr_db=snr)
        rx_bits, H_est_acc, tx_syms_acc, rx_syms_acc = ofdm.transmit_bits(bits)

        min_len = min(len(bits), len(rx_bits))
        bit_errors = int(np.sum(bits[:min_len] != rx_bits[:min_len]))
        ber = bit_errors / len(bits)

        n_bytes = shape[0] * shape[1] * shape[2]
        bits_needed = n_bytes * 8
        if len(rx_bits) < bits_needed:
            rx_bits = np.pad(rx_bits, (0, bits_needed - len(rx_bits)))
        img_rx_arr = np.packbits(rx_bits[:bits_needed]).reshape(shape)
        img_recv = Image.fromarray(img_rx_arr.astype(np.uint8))

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(11, 7))
        fig.patch.set_facecolor("#f8f9fa")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        data_freqs = ofdm.data_freqs_khz
        sort_idx = np.argsort(data_freqs)

        H_real_full = np.abs(ofdm.H_real_channels[0])
        H_est_mean = np.mean(np.abs(np.array(H_est_acc)), axis=0)[0]

        ax1.plot(data_freqs[sort_idx], H_real_full[ofdm.data_sc][sort_idx], label="Canal real |H_1(f)|", color="#0055ff", lw=1.5)
        ax1.plot(data_freqs[sort_idx], H_est_mean[sort_idx], label="Est. MRC |Ĥ_1(f)|", color="#ff5500", ls="--", lw=1.2)
        ax1.set_title(f"Canal en Frecuencia (Antena 1) ", fontweight="bold", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(img)
        ax2.set_title("Imagen Original", fontweight="bold", fontsize=10)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(img_recv)
        ax3.set_title(f"Recibida (Rx Antenas: {n_rx})", fontweight="bold", fontsize=10)
        ax3.axis("off")

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

    def run_select_montecarlo(self, target_mod):
        if not self.img_path:
            messagebox.showwarning("Advertencia",
                "Cargue una imagen en la pestaña 'Transmisión General / Imagen' antes de ejecutar el análisis.")
            return

        self.btn_mc_qpsk.config(state="disabled")
        self.btn_mc_16qam.config(state="disabled")
        self.btn_mc_64qam.config(state="disabled")

        iterations = MC_ITERATIONS
        bw = MC_BW_MHZ
        cp = MC_CP_TYPE
        #taps = MC_N_TAPS

        img = Image.open(self.img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        tx_bits_image = np.unpackbits(arr.flatten())
        n_bits_image = len(tx_bits_image)

        snr_axis = np.arange(0, 26, 2.5)
        antenas_test = [1, 4, 8]
        modulations = ["QPSK", "16QAM", "64QAM"]

        total_steps = len(antenas_test) * len(snr_axis) * iterations
        self.mc_progress.config(maximum=total_steps, value=0)
        step_count = 0
        t_start_proc = time.perf_counter()

        colors_rx = {1: "#0055ff", 4: "#ff5500", 8: "#27ae60"}

        for n_rx in antenas_test:
            if not self.running: break # <-- Interrumpir si se cerró la ventana
            total_errors = np.zeros(len(snr_axis), dtype=np.int64)
            total_bits = np.zeros(len(snr_axis), dtype=np.int64)

            for it in range(iterations):
                if not self.running: break # <-- Interrumpir si se cerró la ventana
                ofdm_sim = SIMO_OFDMSystem(bw_mhz=bw, mod=target_mod, cp_type=cp, n_rx=n_rx, snr_db=0)

                for snr_idx, snr in enumerate(snr_axis):
                    if not self.running: break # <-- Interrumpir si se cerró la ventana
                    
                    t_elapsed = time.perf_counter() - t_start_proc
                    pasos_hechos = max(step_count, 1)
                    t_restante = (t_elapsed / pasos_hechos) * (total_steps - step_count)
                    
                    self.lbl_mc_status.config(
                        text=f"Procesando: {target_mod} | Rx Antenas: {n_rx} | SNR: {snr} dB | Iter: {it+1}/{iterations} \n"
                             f"Tiempo Transcurrido: {t_elapsed/60:.2f} min | Est. Restante: {t_restante/60:.2f} min")
                    self.mc_progress.config(value=step_count)
                    
                    # FIX CRÍTICO: .update() le devuelve momentáneamente el control a Tkinter
                    # permitiendo redibujar las ventanas del OS y quitando el molesto cuelgue.
                    self.root.update()

                    ofdm_sim.snr_db = snr
                    rx_bits, _, _, _ = ofdm_sim.transmit_bits(tx_bits_image, lite_mode=True)

                    min_l = min(n_bits_image, len(rx_bits))
                    total_errors[snr_idx] += int(np.sum(tx_bits_image[:min_l] != rx_bits[:min_l]))
                    total_bits[snr_idx] += min_l
                    del rx_bits
                    step_count += 1

                del ofdm_sim
                gc.collect()

            ber_results = np.where(total_bits > 0, total_errors / total_bits, 1.0)
            self.mc_history[target_mod][n_rx] = (snr_axis, ber_results)

        # Re-renderizado dinámico con límites independientes por eje
        for idx, mod_name in enumerate(modulations):
            ax = self.axes_mc[idx]
            
            # Solo actualizamos el gráfico si tiene datos históricos para evitar borrar los otros
            if len(self.mc_history[mod_name]) > 0:
                ax.clear() # Limpiamos de verdad el eje anterior
                ax.set_facecolor("#fbfcfe")
                ax.set_title(f"Modulación {mod_name}", fontweight="bold", fontsize=10)
                ax.set_xlabel("SNR (dB)", fontsize=8)
                ax.set_ylabel("BER (Bit Error Rate)", fontsize=8)
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.set_yscale('log')
                ax.set_xlim(0, 25)

                all_y_values = []
                for rx_config, data_curve in self.mc_history[mod_name].items():
                    x_data, y_data = data_curve
                    ax.semilogy(x_data, y_data, 'o-', label=f"{rx_config} Antenas Rx", color=colors_rx[rx_config], lw=1.5)
                    all_y_values.extend(y_data[y_data > 0]) # Filtrar ceros absolutos

                # Ajuste dinámico e independiente del eje Y
                if len(all_y_values) > 0:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    ax.set_ylim(max(1e-6, y_min * 0.2), min(1.2, y_max * 2))
                else:
                    ax.set_ylim(1e-5, 1.2)

                ax.legend(fontsize=8, loc="upper right")

        # Forzar el refresco estricto del canvas en la UI
        self.fig_mc.canvas.draw_idle() 
        self.root.update_idletasks()

        self.lbl_mc_status.config(text=f"¡Simulación completada y gráfica actualizada para {target_mod}!", foreground="#27ae60")
        self.mc_progress.config(value=total_steps)
        
        self.btn_mc_qpsk.config(state="normal")
        self.btn_mc_16qam.config(state="normal")
        self.btn_mc_64qam.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()