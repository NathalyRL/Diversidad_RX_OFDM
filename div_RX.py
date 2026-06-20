import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─────────────────────────────────────────────
#  TABLAS LTE Y CONFIGURACIONES   (SIN CAMBIOS)
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
#  CANAL MULTIPATH PARA MÚLTIPLES ANTENAS (SIMO)   (SIN CAMBIOS)
# ─────────────────────────────────────────────
def make_simo_channels(fft_size, n_rx, n_taps=6):
    H_channels = []
    h_channels = []
    for i in range(n_rx):
        delays = np.sort(np.random.randint(0, max(1, fft_size//16), n_taps))
        gains  = np.random.randn(n_taps) + 1j * np.random.randn(n_taps)
        gains /= np.sqrt(np.sum(np.abs(gains)**2))
        h = np.zeros(fft_size, dtype=complex)
        for d, g in zip(delays, gains):
            h[d] += g
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
#  OFDM CORE CON DIVERSIDAD Y MRC   (SIN CAMBIOS)
# ─────────────────────────────────────────────
class SIMO_OFDMSystem:

    def regenerate_channel(self):
        """Genera canales nuevos para la diversidad en Montecarlo."""
        self.H_real_channels, self.h_channels = make_simo_channels(
            self.fft_size, self.n_rx, self.n_taps
        )

    def interleave(self, bits):
        # Genera un patrón de permutación aleatorio para romper la correlación frecuencial
        np.random.seed(42) 
        indices = np.arange(len(bits))
        np.random.shuffle(indices)
        return bits[indices], indices

    def deinterleave(self, bits, indices):
        restored = np.zeros_like(bits)
        restored[indices] = bits
        return restored
    
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

        self.H_real_channels, self.h_channels = make_simo_channels(self.fft_size, self.n_rx, self.n_taps)

    def modulate(self, data_symbols, pilot_value=1.0+0j):
        frame = np.zeros(self.fft_size, dtype=complex)
        frame[self.pilot_sc] = pilot_value
        frame[self.data_sc] = data_symbols
        time_sig = np.fft.ifft(frame) * np.sqrt(self.fft_size)
        cp = time_sig[-self.cp_len:]
        return np.concatenate([cp, time_sig])

    def demodulate_and_estimate(self, rx_signals_list, pilot_value=1.0+0j):
        # Listas para guardar las señales en frecuencia y las estimaciones por antena
        data_rx_all = []
        H_est_data_all = []
        H_pilots_all = []

        for rx_sig in rx_signals_list:
            ofdm_sym = rx_sig[self.cp_len:]
            freq = np.fft.fft(ofdm_sym) / np.sqrt(self.fft_size)
            pilots_rx = freq[self.pilot_sc]
            data_rx = freq[self.data_sc]

            # Estimación por interpolación
            H_pilots = pilots_rx / pilot_value
            H_est_data = np.interp(self.data_idx_local, self.pilot_idx_local, H_pilots.real) + \
                         1j * np.interp(self.data_idx_local, self.pilot_idx_local, H_pilots.imag)

            data_rx_all.append(data_rx)
            H_est_data_all.append(H_est_data)
            H_pilots_all.append(H_pilots)

        # ── COMBINACIÓN MAXIMAL RATIO COMBINING (MRC) ──
        # Numerador: suma(H_est* * Y_rx), Denominador: suma(|H_est|^2)
        num_mrc = np.zeros_like(data_rx_all[0], dtype=complex)
        den_mrc = np.zeros_like(data_rx_all[0], dtype=float)

        for i in range(self.n_rx):
            num_mrc += np.conj(H_est_data_all[i]) * data_rx_all[i]
            den_mrc += np.abs(H_est_data_all[i])**2

        eq_data = num_mrc / (den_mrc + 1e-10)
        return eq_data, H_est_data_all

    def transmit_bits(self, bits):
        bits_shuffled, indices = self.interleave(bits)
        syms_data, pad = bits_to_symbols(bits_shuffled, self.constellation, self.bps)
        n_ofdm = int(np.ceil(len(syms_data) / self.n_data_per_sym))

        all_rx_bits = []
        H_est_acc = []
        tx_syms_acc = []
        rx_syms_acc = []

        for i in range(n_ofdm):
            chunk = syms_data[i*self.n_data_per_sym : (i+1)*self.n_data_per_sym]
            if len(chunk) < self.n_data_per_sym:
                chunk = np.pad(chunk, (0, self.n_data_per_sym - len(chunk)))

            tx_signal = self.modulate(chunk)
            rx_signals_simo = apply_simo_channel(tx_signal, self.h_channels, self.snr_db)

            eq_data, H_est_data_all = self.demodulate_and_estimate(rx_signals_simo)

            rx_bits_sym = symbols_to_bits(eq_data, self.constellation, self.bps)
            all_rx_bits.append(rx_bits_sym)
            H_est_acc.append(H_est_data_all)
            tx_syms_acc.append(chunk)
            rx_syms_acc.append(eq_data)

        all_rx_bits = np.concatenate(all_rx_bits)

        all_rx_bits = all_rx_bits[:len(bits)]
        if pad > 0: all_rx_bits = all_rx_bits[:-pad]
        
        final_bits = self.deinterleave(all_rx_bits, indices)
        
        return final_bits, H_est_acc, tx_syms_acc, rx_syms_acc


# ─────────────────────────────────────────────
#  PALETA DE COLORES Y TIPOGRAFÍA DE LA INTERFAZ
# ─────────────────────────────────────────────
COLOR_BG_DARK     = "#16213e"
COLOR_BG_DARKER   = "#0f1729"
COLOR_BG_LIGHT    = "#eef1f7"
COLOR_CARD        = "#ffffff"
COLOR_BORDER      = "#dde3ee"
COLOR_ACCENT      = "#3a86ff"
COLOR_ACCENT_DARK = "#2667cc"
COLOR_ACCENT_SOFT = "#e8f0fe"
COLOR_SUCCESS     = "#27ae60"
COLOR_DANGER      = "#e74c3c"
COLOR_WARNING     = "#f39c12"
COLOR_TEXT_DARK   = "#23272f"
COLOR_TEXT_MUTED  = "#6c7689"
COLOR_TEXT_LIGHT  = "#f5f7fa"
FONT_FAMILY       = "Segoe UI"

# Colores usados únicamente para el estilo de las gráficas matplotlib
PLOT_PRIMARY   = COLOR_ACCENT
PLOT_SECONDARY = "#ff6b35"
PLOT_TERTIARY  = COLOR_SUCCESS
PLOT_TEXT      = COLOR_TEXT_DARK
PLOT_PANEL_BG  = COLOR_ACCENT_SOFT


# ─────────────────────────────────────────────
#  INTERFAZ GRÁFICA TKINTER  (REDISEÑADA)
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM · SIMO con MRC")
        self.root.geometry("1280x830")
        self.root.minsize(960, 600)
        self.root.configure(bg=COLOR_BG_LIGHT)

        self.img_path = ""
        self.img_preview = None

        self._build_style()
        self._build_header()

        # Notebook principal para las dos pestañas
        self.notebook = ttk.Notebook(root, style="App.TNotebook")
        self.notebook.pack(fill='both', expand=True, padx=14, pady=(0, 8))

        self.tab_main = ttk.Frame(self.notebook, style="Body.TFrame")
        self.tab_montecarlo = ttk.Frame(self.notebook, style="Body.TFrame")

        self.notebook.add(self.tab_main, text="  🖼️  Transmisión General / Imagen  ")
        self.notebook.add(self.tab_montecarlo, text="  📊  Simulación Montecarlo  ")

        self.setup_tab_main()
        self.setup_tab_montecarlo()

        self._build_statusbar()

    # ───────────────────────────────────────
    #  ESTILO VISUAL GLOBAL
    # ───────────────────────────────────────
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure(".", font=(FONT_FAMILY, 10), background=COLOR_BG_LIGHT)
        style.configure("Body.TFrame", background=COLOR_BG_LIGHT)
        style.configure("Sidebar.TFrame", background=COLOR_BG_LIGHT)

        style.configure("Card.TFrame", background=COLOR_CARD, relief="solid",
                         borderwidth=1, bordercolor=COLOR_BORDER)

        style.configure("Card.TLabelframe", background=COLOR_CARD, relief="solid",
                         borderwidth=1, bordercolor=COLOR_BORDER)
        style.configure("Card.TLabelframe.Label", background=COLOR_CARD,
                         foreground=COLOR_ACCENT_DARK, font=(FONT_FAMILY, 10, "bold"))

        style.configure("TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_DARK, font=(FONT_FAMILY, 10))
        style.configure("Hint.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_MUTED, font=(FONT_FAMILY, 9))
        style.configure("CardTitle.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_DARK,
                         font=(FONT_FAMILY, 12, "bold"))

        style.configure("TEntry", padding=6)
        style.configure("TSpinbox", padding=6, arrowsize=14)
        style.configure("TCombobox", padding=6)

        style.configure("Accent.TButton", font=(FONT_FAMILY, 10, "bold"),
                         background=COLOR_ACCENT, foreground="white", padding=10, borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", COLOR_ACCENT_DARK), ("disabled", "#a9b6c9")],
                  foreground=[("disabled", "#eef1f7")])

        style.configure("Secondary.TButton", font=(FONT_FAMILY, 9, "bold"),
                         background=COLOR_BG_LIGHT, foreground=COLOR_ACCENT_DARK, padding=8, borderwidth=1)
        style.map("Secondary.TButton", background=[("active", COLOR_ACCENT_SOFT)])

        style.configure("App.TNotebook", background=COLOR_BG_LIGHT, borderwidth=0, tabmargins=[6, 6, 6, 0])
        style.configure("App.TNotebook.Tab", background=COLOR_BG_LIGHT, foreground=COLOR_TEXT_MUTED,
                         font=(FONT_FAMILY, 10, "bold"), padding=[14, 9], borderwidth=0)
        style.map("App.TNotebook.Tab",
                  background=[("selected", COLOR_CARD)],
                  foreground=[("selected", COLOR_ACCENT_DARK)])

        style.configure("Horizontal.TProgressbar", troughcolor=COLOR_BG_LIGHT,
                         background=COLOR_ACCENT, borderwidth=0, thickness=14)

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLOR_BG_DARK, height=72)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        icon_canvas = tk.Canvas(header, width=56, height=56, bg=COLOR_BG_DARK, highlightthickness=0)
        icon_canvas.pack(side='left', padx=(22, 10), pady=8)
        # Ícono estilizado: antena emitiendo señal
        icon_canvas.create_oval(8, 38, 48, 50, fill=COLOR_ACCENT_DARK, outline="")
        icon_canvas.create_line(28, 38, 28, 16, fill=COLOR_TEXT_LIGHT, width=3)
        icon_canvas.create_oval(22, 8, 34, 20, fill=COLOR_ACCENT, outline="")
        icon_canvas.create_arc(9, 1, 47, 39, start=25, extent=130, style="arc",
                                outline=COLOR_ACCENT, width=2)
        icon_canvas.create_arc(0, -8, 56, 48, start=25, extent=130, style="arc",
                                outline="#5d6a85", width=2)

        title_frame = tk.Frame(header, bg=COLOR_BG_DARK)
        title_frame.pack(side='left', pady=10)
        tk.Label(title_frame, text="Simulador OFDM", bg=COLOR_BG_DARK,
                 fg=COLOR_TEXT_LIGHT, font=(FONT_FAMILY, 16, "bold")).pack(anchor='w')
        tk.Label(title_frame, text="Diversidad SIMO  ·  Combinación de Máxima Razón (MRC)",
                 bg=COLOR_BG_DARK, fg="#9aa7c2", font=(FONT_FAMILY, 9)).pack(anchor='w')

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=COLOR_BG_DARKER, height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Listo")
        tk.Label(bar, textvariable=self.status_var, bg=COLOR_BG_DARKER, fg="#9aa7c2",
                 font=(FONT_FAMILY, 8), anchor='w').pack(side='left', padx=14)
        tk.Label(bar, text="LTE OFDM · MRC Diversity Lab", bg=COLOR_BG_DARKER, fg="#5d6a85",
                 font=(FONT_FAMILY, 8)).pack(side='right', padx=14)

    # ───────────────────────────────────────
    #  UTILIDAD: scroll con rueda del mouse (solo mientras el cursor
    #  está sobre el panel correspondiente, para no interferir con
    #  otros widgets desplazables de la ventana)
    # ───────────────────────────────────────
    def _enable_mousewheel(self, canvas):
        def _on_wheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-2, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(2, "units")
            elif getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 40)), "units")

        def _activate(_event):
            canvas.bind_all("<MouseWheel>", _on_wheel)
            canvas.bind_all("<Button-4>", _on_wheel)
            canvas.bind_all("<Button-5>", _on_wheel)

        def _deactivate(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _activate)
        canvas.bind("<Leave>", _deactivate)

    # ───────────────────────────────────────
    #  PESTAÑA 1: TRANSMISIÓN GENERAL / IMAGEN
    # ───────────────────────────────────────
    def setup_tab_main(self):
        self.tab_main.columnconfigure(0, weight=0)
        self.tab_main.columnconfigure(1, weight=1)
        self.tab_main.rowconfigure(0, weight=1)

        # ── Barra lateral de parámetros ──
        sidebar_outer = ttk.Frame(self.tab_main, style="Sidebar.TFrame")
        sidebar_outer.grid(row=0, column=0, sticky='ns', padx=(14, 8), pady=14)
        sidebar_outer.rowconfigure(0, weight=1)
        sidebar_outer.columnconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_outer, bg=COLOR_BG_LIGHT, highlightthickness=0, width=272)
        sidebar_scroll = ttk.Scrollbar(sidebar_outer, orient='vertical', command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.grid(row=0, column=0, sticky='ns')
        sidebar_scroll.grid(row=0, column=1, sticky='ns')

        sidebar = ttk.Frame(sidebar_canvas, style="Sidebar.TFrame")
        sidebar_window_id = sidebar_canvas.create_window((0, 0), window=sidebar, anchor='nw')

        def _update_scrollregion(event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox('all'))
        sidebar.bind('<Configure>', _update_scrollregion)

        def _match_canvas_width(event):
            sidebar_canvas.itemconfig(sidebar_window_id, width=event.width)
        sidebar_canvas.bind('<Configure>', _match_canvas_width)

        self._enable_mousewheel(sidebar_canvas)

        card_ant = ttk.LabelFrame(sidebar, text="  📡  Antenas Receptoras  ",
                                   style="Card.TLabelframe", padding=14)
        card_ant.pack(fill='x', pady=(0, 12))
        ttk.Label(card_ant, text="Número de antenas Rx:", style="TLabel").grid(
            row=0, column=0, sticky='w', pady=4)
        self.spin_rx = ttk.Spinbox(card_ant, from_=1, to=16, width=10)
        self.spin_rx.set(2)
        self.spin_rx.grid(row=0, column=1, pady=4, padx=(8, 0))
        ttk.Label(card_ant, text="Más antenas → mayor ganancia de diversidad",
                  style="Hint.TLabel", wraplength=220).grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(4, 0))

        card_mod = ttk.LabelFrame(sidebar, text="  🎚️  Modulación y Canal  ",
                                   style="Card.TLabelframe", padding=14)
        card_mod.pack(fill='x', pady=(0, 12))

        ttk.Label(card_mod, text="Modulación:", style="TLabel").grid(row=0, column=0, sticky='w', pady=4)
        self.combo_mod = ttk.Combobox(card_mod, values=["QPSK", "16QAM", "64QAM"], state="readonly", width=11)
        self.combo_mod.set("16QAM")
        self.combo_mod.grid(row=0, column=1, pady=4, padx=(8, 0))

        ttk.Label(card_mod, text="Ancho de banda (MHz):", style="TLabel").grid(row=1, column=0, sticky='w', pady=4)
        self.combo_bw = ttk.Combobox(card_mod, values=["1.4", "3.0", "5.0", "10.0", "15.0", "20.0"],
                                      state="readonly", width=11)
        self.combo_bw.set("5.0")
        self.combo_bw.grid(row=1, column=1, pady=4, padx=(8, 0))

        ttk.Label(card_mod, text="Prefijo cíclico:", style="TLabel").grid(row=2, column=0, sticky='w', pady=4)
        self.combo_cp = ttk.Combobox(card_mod, values=["normal", "extendido"], state="readonly", width=11)
        self.combo_cp.set("normal")
        self.combo_cp.grid(row=2, column=1, pady=4, padx=(8, 0))

        ttk.Label(card_mod, text="Taps del canal:", style="TLabel").grid(row=3, column=0, sticky='w', pady=4)
        self.spin_taps = ttk.Spinbox(card_mod, from_=1, to=20, width=10)
        self.spin_taps.set(6)
        self.spin_taps.grid(row=3, column=1, pady=4, padx=(8, 0))

        card_snr = ttk.LabelFrame(sidebar, text="  ⚡  Condiciones de Prueba  ",
                                   style="Card.TLabelframe", padding=14)
        card_snr.pack(fill='x', pady=(0, 12))
        ttk.Label(card_snr, text="SNR de prueba (dB):", style="TLabel").grid(row=0, column=0, sticky='w', pady=4)
        self.entry_snr = ttk.Entry(card_snr, width=13)
        self.entry_snr.insert(0, "15")
        self.entry_snr.grid(row=0, column=1, pady=4, padx=(8, 0))

        card_img = ttk.LabelFrame(sidebar, text="  🖼️  Imagen de Entrada  ",
                                   style="Card.TLabelframe", padding=14)
        card_img.pack(fill='x', pady=(0, 12))
        ttk.Button(card_img, text="Seleccionar Imagen", style="Secondary.TButton",
                   command=self.load_image).pack(fill='x', pady=(0, 8))
        self.lbl_img = ttk.Label(card_img, text="Sin imagen cargada", style="Hint.TLabel",
                                  foreground=COLOR_DANGER)
        self.lbl_img.pack(anchor='w')
        self.lbl_preview = tk.Label(card_img, bg=COLOR_CARD, text="(vista previa)",
                                     fg=COLOR_TEXT_MUTED, font=(FONT_FAMILY, 8))
        self.lbl_preview.pack(pady=(8, 0))

        ttk.Button(sidebar, text="▶  Simular Transmisión", style="Accent.TButton",
                   command=self.run_main_simulation).pack(fill='x', pady=(4, 0))

        # ── Panel de resultados ──
        result_card = ttk.Frame(self.tab_main, style="Card.TFrame")
        result_card.grid(row=0, column=1, sticky='nsew', padx=(8, 14), pady=14)
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(2, weight=1)

        accent_bar = tk.Frame(result_card, bg=COLOR_ACCENT, height=4)
        accent_bar.grid(row=0, column=0, sticky='ew')

        header = ttk.Frame(result_card, style="Card.TFrame")
        header.grid(row=1, column=0, sticky='ew', padx=18, pady=(14, 0))
        ttk.Label(header, text="Resultados de la Simulación", style="CardTitle.TLabel").pack(anchor='w')

        self.plot_frame = ttk.Frame(result_card, style="Card.TFrame")
        self.plot_frame.grid(row=2, column=0, sticky='nsew', padx=18, pady=16)

        self.placeholder_main = ttk.Label(
            self.plot_frame,
            text="👈  Configure los parámetros y presione\n“Simular Transmisión” para ver los resultados",
            style="Hint.TLabel", justify='center', font=(FONT_FAMILY, 11)
        )
        self.placeholder_main.pack(expand=True)

    # ───────────────────────────────────────
    #  PESTAÑA 2: SIMULACIÓN MONTECARLO
    # ───────────────────────────────────────
    def setup_tab_montecarlo(self):
        self.tab_montecarlo.columnconfigure(0, weight=1)
        self.tab_montecarlo.rowconfigure(1, weight=1)

        config_card = ttk.Frame(self.tab_montecarlo, style="Card.TFrame")
        config_card.grid(row=0, column=0, sticky='ew', padx=14, pady=14)
        config_card.columnconfigure(0, weight=1)

        accent_bar = tk.Frame(config_card, bg=COLOR_ACCENT, height=4)
        accent_bar.pack(fill='x')

        inner = ttk.Frame(config_card, style="Card.TFrame", padding=16)
        inner.pack(fill='x')
        inner.columnconfigure(3, weight=1)

        ttk.Label(inner, text="Configuración de Simulación Estocástica", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky='w', pady=(0, 4))
        ttk.Label(inner, text="Calcula curvas de BER vs SNR para QPSK, 16QAM y 64QAM con 1, 3 y 8 antenas Rx.",
                  style="Hint.TLabel").grid(row=1, column=0, columnspan=4, sticky='w', pady=(0, 12))

        ttk.Label(inner, text="Número de Iteraciones:", style="TLabel").grid(
            row=2, column=0, sticky='w', padx=(0, 8))
        self.entry_iter = ttk.Entry(inner, width=12)
        self.entry_iter.insert(0, "100")
        self.entry_iter.grid(row=2, column=1, sticky='w', padx=(0, 20))

        self.btn_run_montecarlo = ttk.Button(inner, text="📈  Ejecutar Curvas BER de Montecarlo",
                                              style="Accent.TButton", command=self.run_montecarlo)
        self.btn_run_montecarlo.grid(row=2, column=2, padx=(0, 20))

        self.mc_progress = ttk.Progressbar(inner, orient='horizontal', length=200, mode='determinate')
        self.mc_progress.grid(row=2, column=3, sticky='ew')

        self.mc_status_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.mc_status_var, style="Hint.TLabel").grid(
            row=3, column=0, columnspan=4, sticky='w', pady=(8, 0))

        result_card = ttk.Frame(self.tab_montecarlo, style="Card.TFrame")
        result_card.grid(row=1, column=0, sticky='nsew', padx=14, pady=(0, 14))
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(0, weight=1)

        self.mc_plot_frame = ttk.Frame(result_card, style="Card.TFrame")
        self.mc_plot_frame.grid(row=0, column=0, sticky='nsew', padx=18, pady=16)

        self.placeholder_mc = ttk.Label(
            self.mc_plot_frame,
            text="📊  Presione “Ejecutar Curvas BER de Montecarlo”\npara generar el análisis estadístico",
            style="Hint.TLabel", justify='center', font=(FONT_FAMILY, 11)
        )
        self.placeholder_mc.pack(expand=True)

    # ───────────────────────────────────────
    #  CARGA DE IMAGEN (con miniatura de vista previa)
    # ───────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.img_path = path
            self.lbl_img.config(text=f"✅ {os.path.basename(path)}", foreground=COLOR_SUCCESS)
            self.status_var.set(f"Imagen cargada: {os.path.basename(path)}")
            try:
                preview = Image.open(path).convert("RGB")
                preview.thumbnail((140, 140))
                self.img_preview = ImageTk.PhotoImage(preview)
                self.lbl_preview.config(image=self.img_preview, text="")
            except Exception:
                pass

    # ───────────────────────────────────────
    #  SIMULACIÓN PRINCIPAL  (lógica original sin cambios)
    # ───────────────────────────────────────
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

        self.status_var.set("Simulando transmisión OFDM...")
        self.root.update_idletasks()

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
        fig.patch.set_facecolor(COLOR_CARD)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

        # 1. Canal Real vs Estimado (Se grafica la antena 1 de ejemplo)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor("#fbfcfe")
        data_freqs = ofdm.data_freqs_khz
        sort_idx = np.argsort(data_freqs)

        H_real_full = np.abs(ofdm.H_real_channels[0])
        H_est_mean = np.mean(np.abs(np.array(H_est_acc)), axis=0)[0]  # Antena 1

        ax1.plot(data_freqs[sort_idx], H_real_full[ofdm.data_sc][sort_idx],
                 label="Canal real |H_1(f)|", color=PLOT_PRIMARY, lw=1.8)
        ax1.plot(data_freqs[sort_idx], H_est_mean[sort_idx],
                 label="Est. MRC |Ĥ_1(f)|", color=PLOT_SECONDARY, ls="--", lw=1.4)
        ax1.set_title(f"Canal en Frecuencia (Antena 1) con {taps} Taps", fontweight="bold",
                      fontsize=10, color=PLOT_TEXT)
        ax1.set_xlabel("Frecuencia (kHz)", fontsize=8, color=PLOT_TEXT)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=8, frameon=False)
        for spine in ax1.spines.values():
            spine.set_color(COLOR_BORDER)

        # 2. Imagen Original
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(img)
        ax2.set_title("Imagen Original", fontweight="bold", fontsize=10, color=PLOT_TEXT)
        ax2.axis("off")

        # 3. Imagen Recibida
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(img_recv)
        ax3.set_title(f"Recibida (Rx Antenas: {n_rx})", fontweight="bold", fontsize=10, color=PLOT_TEXT)
        ax3.axis("off")

        # 4. Datos Estadísticos
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        ax4.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax4.transAxes,
                                     facecolor=PLOT_PANEL_BG, edgecolor=COLOR_BORDER, lw=1, zorder=0))
        stats = [
            ("🔧", "Modulación", mod),
            ("📶", "Ancho Banda", f"{bw} MHz"),
            ("🧮", "FFT size", str(ofdm.fft_size)),
            ("⚡", "SNR", f"{snr} dB"),
            ("❌", "Errores bit", str(bit_errors)),
            ("📉", "BER", f"{ber:.4e}"),
        ]
        y_pos = 0.84
        for icon, label, value in stats:
            ax4.text(0.12, y_pos, f"{icon}  {label}", fontsize=9, fontweight="bold", color=PLOT_TEXT)
            ax4.text(0.12, y_pos - 0.07, value, fontsize=10, color=COLOR_ACCENT_DARK, fontweight="bold")
            y_pos -= 0.18

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.status_var.set(f"Simulación completada — BER = {ber:.4e}")

    # ───────────────────────────────────────
    #  SIMULACIÓN MONTECARLO  (lógica original sin cambios)
    # ───────────────────────────────────────
    def run_montecarlo(self):
        try:
            iterations = int(self.entry_iter.get())
        except ValueError:
            messagebox.showerror("Error", "Coloque un número válido de iteraciones.")
            return

        self.btn_run_montecarlo.config(state="disabled")
        self.root.update_idletasks()

        snr_axis = np.arange(0, 26, 5)
        antenas_test = [1, 3, 8]
        modulations = ["QPSK", "16QAM", "64QAM"]

        total_steps = len(modulations) * len(antenas_test) * len(snr_axis)
        self.mc_progress['maximum'] = total_steps
        self.mc_progress['value'] = 0
        step_count = 0

        # Limpiar área de gráficas de montecarlo
        for widget in self.mc_plot_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        fig.patch.set_facecolor(COLOR_CARD)

        colors_rx = {1: PLOT_PRIMARY, 3: PLOT_SECONDARY, 8: PLOT_TERTIARY}

        # Loop Montecarlo simulando transmisiones de ráfagas aleatorias por cada configuración
        for m_idx, mod in enumerate(modulations):
            bps = BITS_PER_SYMBOL[mod]
            ax = axes[m_idx]
            ax.set_facecolor("#fbfcfe")

            for n_rx in antenas_test:
                ber_results = []
                for snr in snr_axis:
                    total_errors = 0
                    total_bits = 0

                    # Generación dinámica del bloque estocástico
                    ofdm_sim = SIMO_OFDMSystem(bw_mhz=5.0, mod=mod, cp_type="normal", n_rx=n_rx, snr_db=snr, n_taps=4)
                    bits_per_iter = ofdm_sim.n_data_per_sym * bps

                    for _ in range(iterations):
                        ofdm_sim.regenerate_channel()
                        tx_bits = np.random.randint(0, 2, bits_per_iter)
                        rx_bits, _, _, _ = ofdm_sim.transmit_bits(tx_bits)

                        min_l = min(len(tx_bits), len(rx_bits))
                        total_errors += np.sum(tx_bits[:min_l] != rx_bits[:min_l])
                        total_bits += len(tx_bits)

                    ber_results.append(total_errors / total_bits if total_bits > 0 else 1)

                    # ── Actualización visual de progreso (no afecta el cálculo) ──
                    step_count += 1
                    self.mc_progress['value'] = step_count
                    self.mc_status_var.set(f"Calculando: {mod}, {n_rx} antenas Rx, SNR = {snr} dB...")
                    self.root.update_idletasks()

                ax.semilogy(snr_axis, ber_results, 'o-', color=colors_rx.get(n_rx, PLOT_PRIMARY),
                            label=f"{n_rx} Antenas Rx", lw=1.8, markersize=5)

            ax.set_title(f"Modulación {mod}", fontweight="bold", fontsize=10, color=PLOT_TEXT)
            ax.set_xlabel("SNR (dB)", fontsize=9, color=PLOT_TEXT)
            if m_idx == 0:
                ax.set_ylabel("BER (Bit Error Rate)", fontsize=9, color=PLOT_TEXT)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8, frameon=False)
            ax.set_ylim(1e-5, 1)
            for spine in ax.spines.values():
                spine.set_color(COLOR_BORDER)

        fig.suptitle("Análisis Estadístico de Montecarlo usando Diversidad MRC",
                      fontweight="bold", fontsize=12, color=PLOT_TEXT)

        canvas = FigureCanvasTkAgg(fig, master=self.mc_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        self.mc_status_var.set(f"Simulación completada — {iterations} iteraciones por punto")
        self.btn_run_montecarlo.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()