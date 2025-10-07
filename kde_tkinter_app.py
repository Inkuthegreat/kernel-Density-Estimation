import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def gauss(x):
    return (np.exp(-(x**2)/2)/np.sqrt(2*np.pi))

def uform(x):
    return (0.5*(np.abs(x)<=1))

def kde(x_points, data, bandwidth, ktype='gaussian'):
    n = len(data)
    density_estimate = np.zeros_like(x_points, dtype=float)
    for i in range(len(x_points)):
        x = x_points[i]
        u = (x - data) / bandwidth
        if ktype == 'gaussian':
            k_value = gauss(u)
        elif ktype == 'uniform':
            k_value = uform(u)
        density_estimate[i] = np.sum(k_value) / (n * bandwidth)
    return density_estimate


class KDEApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Panel")
        self.geometry("1200x720")
        self.configure(bg='#f8f9fc')

        # Sidebar
        self.sidebar = tk.Frame(self, bg='#343a40', width=200)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(0)

        # Topbar/Header
        self.topbar = tk.Frame(self, bg='#4e73df', height=60)
        self.topbar.pack(side='top', fill='x')
        self.topbar.pack_propagate(0)
        tk.Label(self.topbar, text="Machine Learning Panel", bg='#4e73df', fg="white", 
                 font=("Segoe UI", 18, "bold")).pack(side='left', padx=30, pady=10)

        # Main Content
        self.content = tk.Frame(self, bg='#f8f9fc')
        self.content.pack(side='right', fill='both', expand=True)

        # Sidebar Buttons
        self.kde_btn = tk.Button(
            self.sidebar, text="KDE", bg='#343a40', fg='white', font=('Segoe UI', 13, 'bold'),
            relief='flat', activebackground='#007bff', activeforeground='white',
            command=self.open_kde_page
        )
        self.kde_btn.pack(fill='x', pady=(30, 10), padx=20)

        # For expansion: Add more navigation buttons below as needed

    def open_kde_page(self):
        # Clear content
        for widget in self.content.winfo_children():
            widget.destroy()

        # Card Frame (rounded, shadow)
        card = tk.Frame(self.content, bg='white', bd=0, relief='flat', highlightthickness=1, highlightbackground='#e3e6f0')
        card.place(relx=0.5, rely=0.5, anchor='c', width=1000, height=700)

        # Split into left (inputs) and right (plot)
        left = tk.Frame(card, bg='white')
        left.place(relx=0.02, rely=0.02, relwidth=0.36, relheight=0.95)
        right = tk.Frame(card, bg='#f8f9fc')
        right.place(relx=0.40, rely=0.02, relwidth=0.57, relheight=0.95)

        # Inputs Card
        tk.Label(left, text="Kernel Density Estimation", bg='white', fg='#4e73df', font=('Segoe UI', 15, 'bold')).pack(pady=(10,12))

        file_var = tk.StringVar()
        tk.Label(left, text="CSV File", bg='white', anchor='w', font=('Segoe UI', 11)).pack(fill='x', padx=16, pady=(8,0))
        file_entry = tk.Entry(left, textvariable=file_var, width=22, font=('Segoe UI', 10))
        file_entry.pack(padx=16, pady=2)
        def browse_file():
            fname = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if fname:
                file_var.set(fname)
        tk.Button(left, text="Browse", command=browse_file, bg='#4e73df', fg='white', font=('Segoe UI', 10),
                 relief='flat').pack(padx=16, pady=2)

        tk.Label(left, text="Bandwidth (h)", bg='white', anchor='w', font=('Segoe UI', 11)).pack(fill='x', padx=16, pady=(14,0))
        h_var = tk.StringVar(value="0.1")
        tk.Entry(left, textvariable=h_var, width=22, font=('Segoe UI', 10)).pack(padx=16, pady=2)

        tk.Label(left, text="Kernel Type", bg='white', anchor='w', font=('Segoe UI', 11)).pack(fill='x', padx=16, pady=(14,0))
        kernel_var = tk.StringVar(value="gaussian")
        ttk.Combobox(left, textvariable=kernel_var, values=["gaussian", "uniform"], state="readonly").pack(padx=16, pady=2)

        tk.Label(left, text="Bins (Histogram)", bg='white', anchor='w', font=('Segoe UI', 11)).pack(fill='x', padx=16, pady=(14,0))
        bins_var = tk.StringVar(value="80")
        tk.Entry(left, textvariable=bins_var, width=22, font=('Segoe UI', 10)).pack(padx=16, pady=2)

        tk.Label(left, text="Second Bandwidth (h2)", bg='white', anchor='w', font=('Segoe UI', 11)).pack(fill='x', padx=16, pady=(14,0))
        h2_var = tk.StringVar(value="0.3")
        tk.Entry(left, textvariable=h2_var, width=22, font=('Segoe UI', 10)).pack(padx=16, pady=2)


        def plot_kde():
            fname = file_var.get()
            try:
                data = np.loadtxt(fname, delimiter=",")
            except Exception as e:
                messagebox.showerror("Error loading CSV", f"Could not load file: {e}")
                return
            try:
                h = float(h_var.get())
                bins = int(bins_var.get())
            except Exception as e:
                messagebox.showerror("Input Error", f"Invalid parameter: {e}")
                return
            ktype = kernel_var.get()
            x_grid = np.linspace(min(data)-1, max(data)+1, 1000)
            # KDE
            kde_values  = kde(x_grid, data, h, ktype)


            # Plot
            fig = plt.Figure(figsize=(8,5), dpi=100)
            ax = fig.add_subplot(111)
            ax.hist(data, bins=bins, density=True, alpha=0.4, color="orange", edgecolor="gray", label="Histogram")
            ax.plot(x_grid, kde_values , color="blue", linewidth=2, label=f"KDE {ktype} (h={h})")
            ax.set_title("Histogram and Kernel Density Estimation")
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
            ax.legend()
            # Clear previous canvas
            for widget in right.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=right)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        tk.Button(left, text="Compute & Plot", command=plot_kde, bg='#226', fg='white', font=('Arial', 11)).pack(padx=20, pady=25)

if __name__ == '__main__':
    KDEApp().mainloop()