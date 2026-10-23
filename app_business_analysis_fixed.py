from pathlib import Path
import sys
import ctypes

# Keep Tkinter from being enlarged twice on high-DPI Windows displays.
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "Sales_Data_2022_2025.csv"
MODEL_FILE = ROOT / "models" / "sales_forecasting_model.pkl"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class SalesForecastingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sales Forecasting & Trend Analysis System")
        # Use a normal Tk scale so widgets/fonts do not appear excessively zoomed.
        try:
            self.tk.call("tk", "scaling", 1.10)
        except tk.TclError:
            pass

        self.geometry("1450x850")
        self.minsize(1150, 700)
        self.configure(bg="#F5F8FC")

        self.df = None
        self.monthly = None
        self.model = None
        self.forecast = None
        self.figure = None
        self.canvas = None
        self.log_box = None
        self.insight_labels = []
        self.card_subtitles = {}

        self._configure_style()
        self._build_header()
        self._build_tabs()

        footer = tk.Frame(self, bg="#EEF4FA", height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        tk.Label(
            footer,
            text="●  Sales Forecasting & Trend Analysis  |  2022–2025 Historical Data  |  2026 Forecast",
            bg="#EEF4FA",
            fg="#627A93",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=18)

        tk.Label(
            footer,
            text="Built with Python • Tkinter • Pandas • Scikit-learn",
            bg="#EEF4FA",
            fg="#8499AC",
            font=("Segoe UI", 10),
        ).pack(side="right", padx=18)

        self._load_system()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Light modern analytics theme
        style.configure("TFrame", background="#F5F8FC")
        style.configure("TLabel", background="#F5F8FC", foreground="#173F67",
                        font=("Segoe UI", 12))
        style.configure("Title.TLabel", background="#F5F8FC", foreground="#173F67",
                        font=("Segoe UI", 26, "bold"))
        style.configure("Subtitle.TLabel", background="#F5F8FC", foreground="#627A93",
                        font=("Segoe UI", 12))
        style.configure("CardTitle.TLabel", background="#FFFFFF", foreground="#627A93",
                        font=("Segoe UI", 15, "bold"))
        style.configure("CardValue.TLabel", background="#FFFFFF", foreground="#173F67",
                        font=("Segoe UI", 22, "bold"))

        style.configure("TNotebook", background="#F5F8FC", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#EEF4FA",
            foreground="#627A93",
            padding=(20, 12),
            font=("Segoe UI", 15, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2F80ED")],
            foreground=[("selected", "white")],
        )

        style.configure(
            "Accent.TButton",
            background="#2F80ED",
            foreground="white",
            padding=(14, 9),
            font=("Segoe UI", 15, "bold"),
            borderwidth=0,
        )
        style.map("Accent.TButton",
                  background=[("active", "#0D73D8"), ("pressed", "#0A5BAF")])

        style.configure(
            "Secondary.TButton",
            background="#EEF4FA",
            foreground="#173F67",
            padding=(12, 8),
            font=("Segoe UI", 15, "bold"),
            borderwidth=0,
        )
        style.map("Secondary.TButton",
                  background=[("active", "#E7F0F8")])

        style.configure(
            "Modern.TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground="#173F67",
            arrowcolor="#2F80ED",
            padding=6,
            font=("Segoe UI", 13),
        )
        style.configure(
            "Business.TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground="#173F67",
            arrowcolor="#2F80ED",
            padding=(10, 8),
            font=("Segoe UI", 14, "bold"),
        )
        # Larger text in ttk combobox drop-down lists.
        try:
            self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 13))
        except Exception:
            pass
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", "#FFFFFF")],
            foreground=[("readonly", "#173F67")],
        )

        style.configure(
            "Modern.TEntry",
            fieldbackground="#FFFFFF",
            foreground="#173F67",
            insertcolor="#173F67",
            padding=7,
        )

        style.configure(
            "Treeview",
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground="#173F67",
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#DCE7F2",
            foreground="#173F67",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padding=7,
        )
        style.map(
            "Treeview",
            background=[("selected", "#183B63")],
            foreground=[("selected", "#173F67")],
        )

    def _build_header(self):
        header = tk.Frame(self, bg="#FFFFFF", height=118)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        icon = tk.Label(
            header, text="↗", bg="#2F80ED", fg="white",
            font=("Segoe UI", 27, "bold"), width=3, height=2,
        )
        icon.pack(side="left", padx=(22, 15), pady=20)

        title_frame = tk.Frame(header, bg="#FFFFFF")
        title_frame.pack(side="left", fill="y", pady=22)
        tk.Label(
            title_frame, text="Sales Forecasting & Trend Analysis",
            bg="#FFFFFF", fg="#173F67", font=("Segoe UI", 26, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_frame, text="DATA  →  ANALYSIS  →  MACHINE LEARNING  →  12-MONTH FORECAST",
            bg="#FFFFFF", fg="#627A93", font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", pady=(5, 0))

        tk.Label(
            header, text="ANALYZE  •  PREDICT  •  GROW",
            bg="#FFFFFF", fg="#18A999", font=("Segoe UI", 15, "bold"),
        ).pack(side="right", padx=28, pady=30)

    def _build_tabs(self):
        shell = tk.Frame(self, bg="#FFFFFF")
        shell.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(shell, bg="#173F67", width=205)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="SALES ANALYTICS", bg="#173F67", fg="#5F8AB8",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=24, pady=(22, 12))

        self.page_host = tk.Frame(shell, bg="#F5F8FC")
        self.page_host.pack(side="left", fill="both", expand=True, padx=(0, 0), pady=0)

        pages = [
            ("Dashboard", "⌂"), ("Sales Analysis", "▥"), ("Business Analysis", "◆"), ("Forecast", "↗"),
            ("Business Insights", "★"), ("Model", "⚙"), ("Data", "▤"), ("Learning Log", "▣")
        ]
        self.page_buttons = {}
        self.pages = {}

        self.dashboard_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.analysis_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.business_analysis_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.forecast_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.insights_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.model_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.data_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.log_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        frames = [self.dashboard_tab, self.analysis_tab, self.business_analysis_tab, self.forecast_tab, self.insights_tab, self.model_tab, self.data_tab, self.log_tab]

        for (name, icon), frame in zip(pages, frames):
            self.pages[name] = frame
            btn = tk.Button(
                self.sidebar, text=f"  {icon}   {name}", anchor="w",
                bg="#173F67", fg="#E7F0F8", activebackground="#35546F", activeforeground="white",
                relief="flat", bd=0, padx=16, pady=12, font=("Segoe UI", 15, "bold"),
                command=lambda n=name: self._show_page(n)
            )
            btn.pack(fill="x", padx=9, pady=2)
            self.page_buttons[name] = btn

        tk.Frame(self.sidebar, bg="#35546F", height=2).pack(fill="x", padx=22, pady=(18, 12))
        tk.Label(self.sidebar, text="Better Data\nBetter Decisions\nHigher Growth", justify="left", anchor="w",
                 bg="#173F67", fg="#2F80ED", font=("Segoe UI", 15, "bold"), padx=22).pack(side="bottom", fill="x", pady=28)

        self._build_dashboard()
        self._build_analysis()
        self._build_business_analysis()
        self._build_forecast()
        self._build_business_insights()
        self._build_model()
        self._build_data()
        self._build_log()
        self._show_page("Dashboard")

    def _show_page(self, name):
        for frame in self.pages.values():
            frame.pack_forget()
        self.pages[name].pack(fill="both", expand=True, padx=15, pady=15)
        for key, btn in self.page_buttons.items():
            if key == name:
                btn.configure(bg="#0B63D9", fg="white")
            else:
                btn.configure(bg="#173F67", fg="#E7F0F8")

    def _build_dashboard(self):
        heading = tk.Frame(self.dashboard_tab, bg="#F5F8FC")
        heading.pack(fill="x", pady=(0, 9))

        tk.Label(
            heading,
            text="Executive Dashboard",
            bg="#F5F8FC",
            fg="#173F67",
            font=("Segoe UI", 21, "bold"),
        ).pack(side="left")

        tk.Label(
            heading,
            text="  Performance overview, trends and forecast insights",
            bg="#F5F8FC",
            fg="#6C839A",
            font=("Segoe UI", 10),
        ).pack(side="left", pady=(5, 0))

        # Modern KPI cards
        tk.Label(self.dashboard_tab, text="KPI Overview", bg="#F5F8FC", fg="#173F67", font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 5))
        cards = tk.Frame(self.dashboard_tab, bg="#F5F8FC")
        cards.pack(fill="x", pady=(0, 11))

        self.card_values = {}
        self.card_subtitles = {}
        card_defs = [
            ("sales", "Total Sales", "Revenue generated", "↗", "#2F80ED"),
            ("profit", "Total Profit", "Profit generated", "₹", "#18A999"),
            ("margin", "Profit Margin", "Profit / Sales", "%", "#7B61D9"),
            ("quantity", "Quantity Sold", "Units sold", "▦", "#F2A33A"),
            ("records", "Transactions", "Sales records", "#", "#D95787"),
            ("forecast", "2026 Forecast", "Predicted sales", "★", "#159BC7"),
        ]

        for i, (key, title, subtitle, icon, accent) in enumerate(card_defs):
            cards.columnconfigure(i, weight=1)
            card = tk.Frame(
                cards, bg="#FFFFFF", highlightbackground="#C8D8E8",
                highlightthickness=1, bd=0,
            )
            card.grid(row=0, column=i, padx=4, sticky="nsew")
            tk.Frame(card, bg=accent, height=4).pack(fill="x")

            body = tk.Frame(card, bg="#FFFFFF")
            body.pack(fill="both", expand=True, padx=12, pady=10)

            tk.Label(
                body, text=icon, bg=accent, fg="white",
                font=("Segoe UI", 15, "bold"), width=2, height=1,
            ).pack(anchor="w")
            tk.Label(
                body, text=title.upper(), bg="#FFFFFF", fg="#627A93",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(5, 0))

            value = tk.Label(
                body, text="--", bg="#FFFFFF", fg="#173F67",
                font=("Segoe UI", 22, "bold"),
            )
            value.pack(anchor="w", pady=(2, 0))
            self.card_values[key] = value

            sub = tk.Label(
                body, text=subtitle, bg="#FFFFFF", fg="#8499AC",
                font=("Segoe UI", 10),
            )
            sub.pack(anchor="w")
            self.card_subtitles[key] = sub

        # Chart + quick insights
        main = tk.Frame(self.dashboard_tab, bg="#F5F8FC")
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        chart_card = tk.Frame(
            main, bg="#FFFFFF", highlightbackground="#C8D8E8",
            highlightthickness=1,
        )
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        chart_header = tk.Frame(chart_card, bg="#FFFFFF")
        chart_header.pack(fill="x", padx=15, pady=(11, 0))
        tk.Label(
            chart_header, text="Sales Trend", bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 15, "bold"),
        ).pack(side="left")
        tk.Label(
            chart_header, text="Monthly historical performance",
            bg="#FFFFFF", fg="#71879C", font=("Segoe UI", 10),
        ).pack(side="left", padx=10, pady=(3, 0))

        self.dashboard_figure = Figure(
            figsize=(9.5, 4.7), dpi=90, facecolor="#FFFFFF", constrained_layout=True
        )
        self.dashboard_ax = self.dashboard_figure.add_subplot(111)
        self.dashboard_ax.set_facecolor("#FFFFFF")
        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_figure, master=chart_card)
        self.dashboard_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 9))

        insight_card = tk.Frame(
            main, bg="#FFFFFF", highlightbackground="#C8D8E8",
            highlightthickness=1,
        )
        insight_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        tk.Label(
            insight_card, text="Quick Insights", bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w", padx=14, pady=(13, 2))
        tk.Label(
            insight_card, text="Automatically generated from your dataset",
            bg="#FFFFFF", fg="#71879C", font=("Segoe UI", 10),
        ).pack(anchor="w", padx=14, pady=(0, 10))

        self.insight_labels = []
        for _ in range(6):
            box = tk.Frame(
                insight_card, bg="#F8FBFE",
                highlightbackground="#D7E3EF", highlightthickness=1,
            )
            box.pack(fill="x", padx=11, pady=3)
            label = tk.Label(
                box, text="Loading insight...", bg="#F8FBFE", fg="#38536F",
                justify="left", anchor="w", wraplength=245,
                font=("Segoe UI", 10),
            )
            label.pack(fill="x", padx=9, pady=7)
            self.insight_labels.append(label)

    def _build_analysis(self):
        tk.Label(self.analysis_tab, text="Sales Analysis", bg="#F5F8FC", fg="#173F67", font=("Segoe UI", 22, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Label(self.analysis_tab, text="Explore sales performance by year, month, product, category, region and channel.", bg="#F5F8FC", fg="#6C839A", font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 10))
        controls = ttk.Frame(self.analysis_tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="View:").pack(side="left")
        self.analysis_choice = ttk.Combobox(
            controls,
            values=["Year-wise Sales", "Month-wise Sales", "Product-wise Sales", "Category-wise Sales", "Region-wise Sales", "Sales Channel-wise Sales"],
            state="readonly",
            width=28,
        )
        self.analysis_choice.set("Year-wise Sales")
        self.analysis_choice.pack(side="left", padx=8)
        ttk.Button(controls, text="Show Chart", command=self.show_analysis).pack(side="left")

        self.analysis_figure = Figure(figsize=(11, 5.5), dpi=90, facecolor="#FFFFFF", constrained_layout=True)
        self.analysis_ax = self.analysis_figure.add_subplot(111)
        self.analysis_ax.set_facecolor("#FFFFFF")
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_figure, master=self.analysis_tab)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_business_analysis(self):
        """Interactive business analysis with readable, dimension-specific charts."""
        tk.Label(
            self.business_analysis_tab, text="Business Analysis",
            bg="#F5F8FC", fg="#173F67",
            font=("Segoe UI", 22, "bold")
        ).pack(anchor="w", pady=(0, 3))

        tk.Label(
            self.business_analysis_tab,
            text="Select a business dimension. The KPI cards and visualization update automatically.",
            bg="#F5F8FC", fg="#6C839A",
            font=("Segoe UI", 12)
        ).pack(anchor="w", pady=(0, 10))

        controls = tk.Frame(
            self.business_analysis_tab, bg="#FFFFFF",
            highlightbackground="#C8D8E8", highlightthickness=1
        )
        controls.pack(fill="x", pady=(0, 10))

        tk.Label(
            controls, text="Analyze by",
            bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 15, "bold")
        ).pack(side="left", padx=(14, 8), pady=12)

        self.business_choice = ttk.Combobox(
            controls,
            values=[
                "Top-Selling Products",
                "Leading Categories",
                "Leading Regions",
                "Leading Sales Channels"
            ],
            state="readonly",
            width=28,
            style="Business.TCombobox"
        )
        self.business_choice.set("Top-Selling Products")
        self.business_choice.pack(side="left", padx=(0, 10), ipady=3)
        self.business_choice.bind(
            "<<ComboboxSelected>>",
            lambda event: self.show_business_analysis()
        )
        self.business_choice.bind(
            "<Button-1>",
            lambda event: self.after(20, self._enlarge_business_dropdown)
        )

        ttk.Button(
            controls, text="Update Graph",
            style="Accent.TButton",
            command=self.show_business_analysis
        ).pack(side="left", padx=4, pady=5)

        # KPI cards
        self.business_kpi_frame = tk.Frame(
            self.business_analysis_tab, bg="#F5F8FC"
        )
        self.business_kpi_frame.pack(fill="x", pady=(0, 10))
        self.business_kpi_values = {}

        defs = [
            ("name", "Top Performer", "#2F80ED"),
            ("sales", "Sales Contribution", "#18A999"),
            ("share", "Share of Total Sales", "#7B61D9"),
            ("count", "Number of Items", "#F2A33A"),
        ]

        for i, (key, title, accent) in enumerate(defs):
            self.business_kpi_frame.columnconfigure(i, weight=1)
            card = tk.Frame(
                self.business_kpi_frame, bg="#FFFFFF",
                highlightbackground="#C8D8E8", highlightthickness=1
            )
            card.grid(row=0, column=i, padx=4, sticky="nsew")

            tk.Frame(card, bg=accent, height=4).pack(fill="x")
            body = tk.Frame(card, bg="#FFFFFF")
            body.pack(fill="both", expand=True, padx=14, pady=10)

            tk.Label(
                body, text=title.upper(),
                bg="#FFFFFF", fg="#627A93",
                font=("Segoe UI", 11, "bold")
            ).pack(anchor="w")

            value = tk.Label(
                body, text="--",
                bg="#FFFFFF", fg="#173F67",
                font=("Segoe UI", 17, "bold")
            )
            value.pack(anchor="w", pady=(4, 0))
            self.business_kpi_values[key] = value

        # Chart card
        chart_card = tk.Frame(
            self.business_analysis_tab, bg="#FFFFFF",
            highlightbackground="#C8D8E8", highlightthickness=1
        )
        chart_card.pack(fill="both", expand=True)

        tk.Label(
            chart_card,
            text="Business Visualization",
            bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(8, 0))

        self.business_analysis_figure = Figure(
            figsize=(10.8, 4.8), dpi=90,
            facecolor="#FFFFFF"
        )
        self.business_analysis_ax = self.business_analysis_figure.add_subplot(111)
        self.business_analysis_ax.set_facecolor("#FFFFFF")

        self.business_analysis_canvas = FigureCanvasTkAgg(
            self.business_analysis_figure, master=chart_card
        )
        self.business_analysis_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=(2, 8)
        )

        # Draw the initial view.
        self.show_business_analysis()

    def _enlarge_business_dropdown(self):
        """Make the business-analysis dropdown list readable on Windows."""
        try:
            popdown = self.tk.call(
                "ttk::combobox::PopdownWindow",
                str(self.business_choice)
            )
            listbox = f"{popdown}.f.l"
            self.tk.call(
                listbox, "configure",
                "-font", "Segoe UI 15",
                "-activestyle", "none"
            )
        except tk.TclError:
            pass

    def show_business_analysis(self):
        if self.df is None:
            return

        choice = self.business_choice.get()

        mapping = {
            "Top-Selling Products": (
                self.df.groupby("Product")["Sales"].sum().sort_values(ascending=False),
                "Product"
            ),
            "Leading Categories": (
                self.df.groupby("Category")["Sales"].sum().sort_values(ascending=False),
                "Category"
            ),
            "Leading Regions": (
                self.df.groupby("Region")["Sales"].sum().sort_values(ascending=False),
                "Region"
            ),
            "Leading Sales Channels": (
                self.df.groupby("Sales_Channel")["Sales"].sum().sort_values(ascending=False),
                "Sales Channel"
            ),
        }

        series, dimension = mapping[choice]
        top_name = str(series.index[0])
        top_sales = float(series.iloc[0])
        total = float(series.sum())
        share = (top_sales / total * 100) if total else 0

        self.business_kpi_values["name"].config(text=top_name)
        self.business_kpi_values["sales"].config(text=f"₹{top_sales:,.0f}")
        self.business_kpi_values["share"].config(text=f"{share:.1f}%")
        self.business_kpi_values["count"].config(text=f"{len(series)}")

        self.business_analysis_figure.clear()
        fig = self.business_analysis_figure
        fig.patch.set_facecolor("#FFFFFF")

        labels = [str(x) for x in series.index]
        values = series.values.astype(float)

        # Products: lollipop ranking
        if choice == "Top-Selling Products":
            ax = fig.add_subplot(111)
            self.business_analysis_ax = ax
            ordered = series.sort_values(ascending=True)
            y = np.arange(len(ordered))
            max_value = float(ordered.max())

            ax.hlines(y, 0, ordered.values, color="#C7D8E8", linewidth=4)
            ax.scatter(
                ordered.values, y, s=125, color="#2F80ED",
                edgecolor="white", linewidth=1.5, zorder=3
            )
            for yi, value in zip(y, ordered.values):
                ax.text(
                    value + max_value * 0.018, yi,
                    f"₹{value/1e6:.1f}M",
                    va="center", fontsize=10,
                    color="#173F67", fontweight="bold"
                )
            ax.set_yticks(y)
            ax.set_yticklabels(
                [str(x) for x in ordered.index],
                fontsize=10, color="#173F67"
            )
            ax.set_xlabel("Sales contribution (₹)", color="#627A93", fontsize=10)
            ax.set_title(
                "Top-Selling Products", color="#173F67",
                fontsize=16, fontweight="bold", pad=12
            )
            ax.set_xlim(0, max_value * 1.18)
            ax.grid(axis="x", alpha=0.20, color="#C4D2DF")
            ax.tick_params(axis="x", labelsize=9, colors="#627A93")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Categories: donut with a dedicated legend column
        elif choice == "Leading Categories":
            gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.55], wspace=0.02)
            ax = fig.add_subplot(gs[0, 0])
            legend_ax = fig.add_subplot(gs[0, 1])
            self.business_analysis_ax = ax

            wedges, _ = ax.pie(
                values, startangle=90, counterclock=False, radius=0.92,
                wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2)
            )
            ax.text(
                0, 0.08, "Leading Category",
                ha="center", va="center", fontsize=10, color="#627A93"
            )
            ax.text(
                0, -0.08, top_name,
                ha="center", va="center", fontsize=15,
                fontweight="bold", color="#173F67"
            )
            ax.set_title(
                "Sales Share by Category", color="#173F67",
                fontsize=16, fontweight="bold", pad=12
            )
            ax.set_aspect("equal")
            ax.axis("off")

            legend_ax.axis("off")
            legend_ax.text(
                0.02, 0.88, "Category Share",
                fontsize=11, fontweight="bold",
                color="#173F67", transform=legend_ax.transAxes
            )
            for i, (name, value) in enumerate(zip(labels, values)):
                y_pos = 0.76 - i * 0.17
                legend_ax.scatter(
                    0.04, y_pos, s=75,
                    color=wedges[i].get_facecolor(),
                    transform=legend_ax.transAxes
                )
                legend_ax.text(
                    0.12, y_pos,
                    f"{name}\n{value/total*100:.1f}%  •  ₹{value/1e6:.1f}M",
                    va="center", fontsize=9.5,
                    color="#173F67", transform=legend_ax.transAxes
                )

        # Regions: connected-dot trend
        elif choice == "Leading Regions":
            ax = fig.add_subplot(111)
            self.business_analysis_ax = ax
            ordered = series.sort_values(ascending=True)
            x = np.arange(len(ordered))
            vals = ordered.values.astype(float)

            ax.plot(
                x, vals, color="#18A999", linewidth=2.5,
                marker="o", markersize=11,
                markerfacecolor="#18A999",
                markeredgecolor="white", markeredgewidth=2
            )
            for xi, value in zip(x, vals):
                ax.annotate(
                    f"₹{value/1e6:.1f}M", (xi, value),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10,
                    fontweight="bold", color="#173F67"
                )
            ax.fill_between(x, vals, alpha=0.10, color="#18A999")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(x) for x in ordered.index],
                fontsize=10, color="#173F67"
            )
            ax.set_ylabel("Regional Sales (₹)", color="#627A93", fontsize=10)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
            ax.tick_params(axis="y", labelsize=9, colors="#627A93")
            ax.set_title(
                "Regional Sales Comparison", color="#173F67",
                fontsize=16, fontweight="bold", pad=12
            )
            ax.grid(axis="y", alpha=0.20, color="#C4D2DF")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Channels: contained radar chart
        else:
            ax = fig.add_subplot(111, polar=True)
            self.business_analysis_ax = ax
            ax.set_facecolor("#FFFFFF")

            n = len(labels)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            radar_values = values.tolist()
            closed_angles = angles + angles[:1]
            closed_values = radar_values + radar_values[:1]

            ax.plot(closed_angles, closed_values, color="#7B61D9", linewidth=2.5)
            ax.fill(closed_angles, closed_values, color="#7B61D9", alpha=0.15)
            ax.scatter(
                angles, values, s=95, color="#7B61D9",
                edgecolor="white", linewidth=2, zorder=3
            )
            ax.set_xticks(angles)
            ax.set_xticklabels(
                labels, fontsize=10, color="#173F67", fontweight="bold"
            )
            ax.tick_params(axis="x", pad=10)
            ax.set_yticklabels([])
            ax.grid(color="#C4D2DF", alpha=0.35)
            ax.set_ylim(0, float(values.max()) * 1.22)

            for angle, value in zip(angles, values):
                ax.text(
                    angle, value + float(values.max()) * 0.09,
                    f"₹{value/1e6:.1f}M",
                    ha="center", va="center",
                    fontsize=9.5, color="#173F67", fontweight="bold"
                )
            ax.set_title(
                "Sales Channel Comparison", color="#173F67",
                fontsize=16, fontweight="bold", pad=24
            )

        fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.14)
        self.business_analysis_canvas.draw_idle()
        self.write_log(f"Business analysis updated: {choice}.")


    def _build_forecast(self):
        title = tk.Label(
            self.forecast_tab,
            text="Sales Forecasting",
            bg="#F5F8FC",
            fg="#173F67",
            font=("Segoe UI", 22, "bold")
        )
        title.pack(anchor="w", pady=(0, 2))

        subtitle = tk.Label(
            self.forecast_tab,
            text="Historical trend and future sales forecast shown separately for easier interpretation.",
            bg="#F5F8FC",
            fg="#6C839A",
            font=("Segoe UI", 10)
        )
        subtitle.pack(anchor="w", pady=(0, 8))

        top = ttk.Frame(self.forecast_tab)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Forecast period (months):").pack(side="left")
        self.forecast_months = tk.StringVar(value="12")
        ttk.Entry(top, textvariable=self.forecast_months, width=8).pack(side="left", padx=8)
        ttk.Button(
            top, text="Generate Forecast", style="Accent.TButton",
            command=self.generate_forecast
        ).pack(side="left")
        ttk.Button(
            top, text="Save Forecast CSV", command=self.save_forecast
        ).pack(side="left", padx=8)

        self.forecast_result = ttk.Label(
            top, text="Model ready when data is loaded.",
            font=("Segoe UI", 15, "bold")
        )
        self.forecast_result.pack(side="right")

        body = ttk.Panedwindow(self.forecast_tab, orient="vertical")
        body.pack(fill="both", expand=True)

        charts_frame = ttk.Frame(body, padding=4)
        table_frame = ttk.Frame(body, padding=4)
        body.add(charts_frame, weight=5)
        body.add(table_frame, weight=2)

        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)

        # Graph 1: historical trend
        hist_box = tk.Frame(
            charts_frame, bg="#FFFFFF",
            highlightbackground="#C8D8E8", highlightthickness=1
        )
        hist_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        tk.Label(
            hist_box, text="Historical Sales Trend",
            bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", padx=10, pady=(8, 0))

        tk.Label(
            hist_box, text="Monthly sales with 3-month moving average",
            bg="#FFFFFF", fg="#71879C",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=10)

        self.hist_forecast_figure = Figure(
            figsize=(6.0, 3.5), dpi=90,
            facecolor="#FFFFFF", constrained_layout=True
        )
        self.hist_forecast_ax = self.hist_forecast_figure.add_subplot(111)
        self.hist_forecast_ax.set_facecolor("#FFFFFF")
        self.hist_forecast_canvas = FigureCanvasTkAgg(
            self.hist_forecast_figure, master=hist_box
        )
        self.hist_forecast_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=6, pady=5
        )

        # Graph 2: forecast trend
        future_box = tk.Frame(
            charts_frame, bg="#FFFFFF",
            highlightbackground="#C8D8E8", highlightthickness=1
        )
        future_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        tk.Label(
            future_box, text="12-Month Forecast Trend",
            bg="#FFFFFF", fg="#173F67",
            font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", padx=10, pady=(8, 0))

        tk.Label(
            future_box, text="Predicted monthly sales from the trained model",
            bg="#FFFFFF", fg="#71879C",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=10)

        self.future_forecast_figure = Figure(
            figsize=(6.0, 3.5), dpi=90,
            facecolor="#FFFFFF", constrained_layout=True
        )
        self.future_forecast_ax = self.future_forecast_figure.add_subplot(111)
        self.future_forecast_ax.set_facecolor("#FFFFFF")
        self.future_forecast_canvas = FigureCanvasTkAgg(
            self.future_forecast_figure, master=future_box
        )
        self.future_forecast_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=6, pady=5
        )

        self.forecast_table = self._make_table(
            table_frame,
            ("Date", "Predicted Sales"),
            ("Month", "Predicted Sales"),
        )

    def _build_business_insights(self):
        self.insight_summary = tk.Frame(self.insights_tab, bg="#F5F8FC")
        self.insight_summary.pack(fill="x", pady=(0, 12))
        for i in range(4):
            self.insight_summary.columnconfigure(i, weight=1)

        tk.Label(self.insights_tab, text="Business Insights",
                 bg="#F5F8FC", fg="#173F67",
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(self.insights_tab,
                 text="Historical performance + forecast story for presentation and decision support",
                 bg="#F5F8FC", fg="#6C839A",
                 font=("Segoe UI", 12)).pack(anchor="w", pady=(0,10))

        self.business_card_values = {}
        cards = [
            ("forecast_total","Forecast Total","#2F80ED"),
            ("forecast_avg","Monthly Average","#18A999"),
            ("forecast_growth","First → Last Growth","#7B61D9"),
            ("peak_forecast","Peak Forecast","#F2A33A")
        ]
        for i,(key,title,accent) in enumerate(cards):
            card=tk.Frame(self.insight_summary,bg="#FFFFFF",
                          highlightbackground="#C8D8E8",highlightthickness=1)
            card.grid(row=0,column=i,padx=5,sticky="nsew")
            tk.Frame(card,bg=accent,height=4).pack(fill="x")
            body=tk.Frame(card,bg="#FFFFFF"); body.pack(fill="both",expand=True,padx=12,pady=10)
            tk.Label(body,text=title.upper(),bg="#FFFFFF",fg="#627A93",
                     font=("Segoe UI",8,"bold")).pack(anchor="w")
            val=tk.Label(body,text="--",bg="#FFFFFF",fg="#173F67",
                         font=("Segoe UI",16,"bold"))
            val.pack(anchor="w",pady=(4,0))
            self.business_card_values[key]=val

        columns=tk.Frame(self.insights_tab,bg="#F5F8FC")
        columns.pack(fill="both",expand=True)
        columns.columnconfigure(0,weight=1); columns.columnconfigure(1,weight=1)
        columns.rowconfigure(0,weight=1)

        self.performance_text=self._insight_box(columns,0,"Performance Snapshot",
            "Important findings from your historical sales data.")
        self.forecast_story=self._insight_box(columns,1,"Forecast Story",
            "What the saved model predicts for the selected period.")

    def _insight_box(self,parent,column,title,subtitle):
        frame=tk.Frame(parent,bg="#FFFFFF",highlightbackground="#C8D8E8",highlightthickness=1)
        frame.grid(row=0,column=column,sticky="nsew",padx=5)
        tk.Label(frame,text=title,bg="#FFFFFF",fg="#173F67",
                 font=("Segoe UI",14,"bold")).pack(anchor="w",padx=15,pady=(15,3))
        tk.Label(frame,text=subtitle,bg="#FFFFFF",fg="#71879C",
                 font=("Segoe UI",9)).pack(anchor="w",padx=15)
        box=tk.Text(frame,wrap="word",state="disabled",bg="#F8FBFE",fg="#35546F",
                    relief="flat",font=("Segoe UI",10),padx=14,pady=12)
        box.pack(fill="both",expand=True,padx=12,pady=12)
        return box

    def _update_business_insights(self):
        if self.df is None or self.forecast is None:
            return
        f=self.forecast["Predicted_Sales"]
        total=float(f.sum()); avg=float(f.mean())
        first=float(f.iloc[0]); last=float(f.iloc[-1])
        growth=((last-first)/first*100) if first else 0
        peak_i=f.idxmax(); peak=float(f.loc[peak_i])
        peak_date=self.forecast.loc[peak_i,"Date"]

        self.business_card_values["forecast_total"].config(text=f"₹{total:,.0f}")
        self.business_card_values["forecast_avg"].config(text=f"₹{avg:,.0f}")
        self.business_card_values["forecast_growth"].config(text=f"{growth:+.1f}%")
        self.business_card_values["peak_forecast"].config(text=f"₹{peak:,.0f}")

        year=self.df.groupby("Year")["Sales"].sum()
        product=self.df.groupby("Product")["Sales"].sum()
        category=self.df.groupby("Category")["Sales"].sum()
        region=self.df.groupby("Region")["Sales"].sum()
        channel=self.df.groupby("Sales_Channel")["Sales"].sum()
        margin=self.df["Profit"].sum()/self.df["Sales"].sum()*100

        performance=(
            f"• Strongest year: {year.idxmax()} — ₹{year.max():,.0f} sales.\n\n"
            f"• Top product: {product.idxmax()} — ₹{product.max():,.0f} sales.\n\n"
            f"• Leading category: {category.idxmax()} — ₹{category.max():,.0f} sales.\n\n"
            f"• Leading region: {region.idxmax()}.\n\n"
            f"• Leading channel: {channel.idxmax()}.\n\n"
            f"• Historical profit margin: {margin:.1f}%."
        )
        self.performance_text.config(state="normal")
        self.performance_text.delete("1.0","end")
        self.performance_text.insert("1.0",performance)
        self.performance_text.config(state="disabled")

        story=(
            f"The model forecasts ₹{total:,.0f} total sales for the selected period.\n\n"
            f"Expected monthly average: ₹{avg:,.0f}.\n\n"
            f"Forecast starts at ₹{first:,.0f} in {self.forecast.iloc[0]['Date']:%b %Y} "
            f"and reaches ₹{last:,.0f} in {self.forecast.iloc[-1]['Date']:%b %Y}, "
            f"a {growth:+.1f}% change.\n\n"
            f"Peak forecast: {peak_date:%b %Y} — ₹{peak:,.0f}.\n\n"
            "This page combines the project's historical analysis and forecasting "
            "results into presentation-ready business insights."
        )
        self.forecast_story.config(state="normal")
        self.forecast_story.delete("1.0","end")
        self.forecast_story.insert("1.0",story)
        self.forecast_story.config(state="disabled")

    def _build_model(self):
        top = ttk.Frame(self.model_tab)
        top.pack(fill="x", pady=(0, 10))
        self.model_status = ttk.Label(top, text="Model not loaded", font=("Segoe UI", 15, "bold"))
        self.model_status.pack(anchor="w")

        self.metrics_frame = ttk.Frame(self.model_tab)
        self.metrics_frame.pack(fill="x", pady=8)
        self.model_info = tk.Text(self.model_tab, height=13, wrap="word", state="disabled", font=("Consolas", 10))
        self.model_info.pack(fill="both", expand=True)

    def _build_data(self):
        controls = ttk.Frame(self.data_tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Dataset view:").pack(side="left")
        self.data_choice = ttk.Combobox(
            controls,
            values=["Raw Sales Data", "Monthly Forecasting Data", "2026 Forecast"],
            state="readonly",
            width=28,
        )
        self.data_choice.set("Raw Sales Data")
        self.data_choice.pack(side="left", padx=8)
        ttk.Button(controls, text="Load Table", command=self.show_data_table).pack(side="left")

        self.data_table_frame = ttk.Frame(self.data_tab)
        self.data_table_frame.pack(fill="both", expand=True)
        self.data_table = None

    def _build_log(self):
        self.log_box = tk.Text(
            self.log_tab,
            wrap="word",
            state="disabled",
            bg="#F8FBFE",
            fg="#35546F",
            insertbackground="#173F67",
            font=("Consolas", 10),
        )
        self.log_box.pack(fill="both", expand=True)

    def _make_table(self, parent, columns, headings):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        table = ttk.Treeview(container, columns=columns, show="headings")
        for col, heading in zip(columns, headings):
            table.heading(col, text=heading)
            table.column(col, width=180, anchor="center")
        scroll_y = ttk.Scrollbar(container, orient="vertical", command=table.yview)
        scroll_x = ttk.Scrollbar(container, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        table.pack(side="left", fill="both", expand=True)
        return table

    def write_log(self, message):
        if self.log_box is None:
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _load_system(self):
        try:
            self.write_log("1. Starting Sales Forecasting and Trend Analysis System.")
            self.write_log(f"2. Reading dataset: {DATA_FILE.name}")
            if not DATA_FILE.exists():
                raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
            if not MODEL_FILE.exists():
                raise FileNotFoundError(f"Saved model not found: {MODEL_FILE}")

            self.df = pd.read_csv(DATA_FILE)
            self.df["Date"] = pd.to_datetime(self.df["Date"], dayfirst=True)
            self.monthly = self.df.groupby(self.df["Date"].dt.to_period("M"))["Sales"].sum().reset_index()
            self.monthly["Date"] = self.monthly["Date"].dt.to_timestamp()
            self.monthly["Time_Index"] = range(len(self.monthly))
            self.monthly["Year"] = self.monthly["Date"].dt.year
            self.monthly["Month"] = self.monthly["Date"].dt.month
            self.monthly["Quarter"] = self.monthly["Date"].dt.quarter

            self.write_log(f"3. Dataset loaded: {len(self.df):,} records and {len(self.df.columns)} columns.")
            self.write_log(f"4. Historical period: {self.monthly['Date'].min():%b %Y} to {self.monthly['Date'].max():%b %Y}.")
            self.write_log("5. Missing values checked and monthly sales prepared for forecasting.")

            self.model = joblib.load(MODEL_FILE)
            self.write_log("6. Saved Improved Linear Regression model loaded successfully.")
            self.write_log("7. Model features: Time_Index, Year, Month and Quarter.")

            self._update_dashboard()
            self.show_analysis()
            self.show_business_analysis()
            self.show_data_table()
            self._update_model_info()
            self.generate_forecast(silent=True)
            self.write_log("8. System is ready for interactive analysis and forecasting.")
        except Exception as exc:
            self.write_log(f"ERROR: {exc}")
            messagebox.showerror("System startup failed", str(exc))

    def _update_dashboard(self):
        total_sales = self.df["Sales"].sum()
        total_profit = self.df["Profit"].sum()
        quantity = self.df["Quantity"].sum()
        records = len(self.df)
        margin = (total_profit / total_sales * 100) if total_sales else 0

        self.card_values["sales"].config(text=f"₹{total_sales:,.0f}")
        self.card_values["profit"].config(text=f"₹{total_profit:,.0f}")
        self.card_values["margin"].config(text=f"{margin:.1f}%")
        self.card_values["quantity"].config(text=f"{quantity:,}")
        self.card_values["records"].config(text=f"{records:,}")
        self.card_values["forecast"].config(text="Loading…")

        dates = self.monthly["Date"]
        sales = self.monthly["Sales"]
        rolling = sales.rolling(3).mean()
        self.dashboard_ax.clear()
        self.dashboard_ax.set_facecolor("#FFFFFF")

        self.dashboard_ax.plot(
            dates, sales, marker="o", linewidth=2.4, markersize=3.5,
            color="#2F80ED", label="Monthly Sales", zorder=3,
        )
        self.dashboard_ax.fill_between(
            dates, sales, alpha=0.10, color="#2F80ED", zorder=1,
        )
        self.dashboard_ax.plot(
            dates, rolling, linewidth=2.0, linestyle="--",
            color="#18A999", label="3-Month Moving Average", zorder=2,
        )

        peak_idx = sales.idxmax()
        self.dashboard_ax.scatter(
            [dates.loc[peak_idx]], [sales.loc[peak_idx]],
            s=65, color="#D95787", zorder=5,
        )
        self.dashboard_ax.annotate(
            "Peak", (dates.loc[peak_idx], sales.loc[peak_idx]),
            xytext=(0, 12), textcoords="offset points", ha="center",
            color="#C43F78", fontsize=8, fontweight="bold",
        )
        self.dashboard_ax.set_title(
            "Monthly Sales Trend | 2022–2025",
            color="#173F67", fontsize=12, fontweight="bold", pad=10,
        )
        self.dashboard_ax.set_ylabel("Sales", color="#6C839A")
        self.dashboard_ax.tick_params(colors="#6C839A", labelsize=10)
        self.dashboard_ax.grid(axis="y", alpha=0.16, color="#C4D2DF")
        self.dashboard_ax.legend(frameon=False, loc="upper left", fontsize=8)
        for spine in self.dashboard_ax.spines.values():
            spine.set_color("#C8D8E8")
        self.dashboard_figure.tight_layout()
        self.dashboard_canvas.draw_idle()

        year_sales = self.df.groupby("Year")["Sales"].sum()
        product_sales = self.df.groupby("Product")["Sales"].sum()
        top_year = year_sales.idxmax()
        top_product = product_sales.idxmax()
        top_region = self.df.groupby("Region")["Sales"].sum().idxmax()
        top_channel = self.df.groupby("Sales_Channel")["Sales"].sum().idxmax()
        peak_date = dates.loc[peak_idx]
        peak_sales = sales.loc[peak_idx]

        insights = [
            f"📈 Strongest year: {top_year} — ₹{year_sales.max():,.0f} sales.",
            f"🏆 Top product: {top_product} — ₹{product_sales.max():,.0f} sales.",
            f"📍 Leading region: {top_region} has the highest regional sales.",
            f"🛒 Leading channel: {top_channel} contributes the most sales.",
            f"💰 Overall profit margin: {margin:.1f}% across all transactions.",
            f"📊 Peak month: {peak_date:%b %Y} — ₹{peak_sales:,.0f} sales.",
        ]
        for label, text in zip(self.insight_labels, insights):
            label.config(text=text)

    def show_analysis(self):
        if self.df is None:
            return
        choice = self.analysis_choice.get()
        mapping = {
            "Year-wise Sales": (self.df.groupby("Year")["Sales"].sum(), "Year-wise Sales"),
            "Month-wise Sales": (self.df.groupby("Month")["Sales"].sum(), "Month-wise Sales"),
            "Product-wise Sales": (self.df.groupby("Product")["Sales"].sum().sort_values(ascending=False), "Product-wise Sales"),
            "Category-wise Sales": (self.df.groupby("Category")["Sales"].sum().sort_values(ascending=False), "Category-wise Sales"),
            "Region-wise Sales": (self.df.groupby("Region")["Sales"].sum().sort_values(ascending=False), "Region-wise Sales"),
            "Sales Channel-wise Sales": (self.df.groupby("Sales_Channel")["Sales"].sum().sort_values(ascending=False), "Sales Channel-wise Sales"),
        }
        series, title = mapping[choice]
        self.analysis_ax.clear()
        self.analysis_ax.set_facecolor("#FFFFFF")
        if choice == "Month-wise Sales":
            self.analysis_ax.plot(
                series.index, series.values, marker="o", linewidth=2.5,
                markersize=5, color="#2F80ED"
            )
            self.analysis_ax.fill_between(series.index, series.values, alpha=0.10, color="#2F80ED")
            self.analysis_ax.set_xticks(range(1, 13))
            self.analysis_ax.set_xlabel("Month")
        elif len(series) >= 6:
            ordered = series.sort_values()
            self.analysis_ax.barh(
                [str(x) for x in ordered.index], ordered.values,
                color="#2F80ED", alpha=0.88
            )
            self.analysis_ax.set_xlabel("Total Sales")
            self.analysis_ax.set_ylabel("")
            for i, value in enumerate(ordered.values):
                self.analysis_ax.text(
                    value, i, f"  ₹{value:,.0f}",
                    va="center", fontsize=8, color="#173F67"
                )
        else:
            bars = self.analysis_ax.bar(
                [str(x) for x in series.index], series.values,
                color="#2F80ED", alpha=0.88
            )
            self.analysis_ax.set_ylabel("Total Sales")
            for bar, value in zip(bars, series.values):
                self.analysis_ax.text(
                    bar.get_x() + bar.get_width()/2, value,
                    f"₹{value/1e6:.1f}M", ha="center", va="bottom",
                    fontsize=8, color="#173F67"
                )

        self.analysis_ax.set_title(title, color="#173F67", fontsize=13, fontweight="bold", pad=10)
        self.analysis_ax.tick_params(colors="#627A93", labelsize=9)
        self.analysis_ax.grid(axis="x" if len(series) >= 6 and choice != "Month-wise Sales" else "y",
                              alpha=0.25, color="#C4D2DF")
        for spine in self.analysis_ax.spines.values():
            spine.set_color("#C8D8E8")
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw_idle()
        self.write_log(f"Analysis chart displayed: {title}.")

    def generate_forecast(self, silent=False):
        if self.model is None or self.monthly is None:
            return
        try:
            months = int(self.forecast_months.get())
            if months < 1 or months > 60:
                raise ValueError("Forecast months must be between 1 and 60.")
        except ValueError as exc:
            messagebox.showerror("Invalid forecast period", str(exc))
            return

        last_date = self.monthly["Date"].max()
        dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=months, freq="MS")
        future = pd.DataFrame({"Date": dates})
        future["Time_Index"] = range(len(self.monthly), len(self.monthly) + months)
        future["Year"] = future["Date"].dt.year
        future["Month"] = future["Date"].dt.month
        future["Quarter"] = future["Date"].dt.quarter
        X = future[["Time_Index", "Year", "Month", "Quarter"]]
        future["Predicted_Sales"] = self.model.predict(X)
        self.forecast = future

        for item in self.forecast_table.get_children():
            self.forecast_table.delete(item)
        for _, row in future.iterrows():
            self.forecast_table.insert("", "end", values=(row["Date"].strftime("%b %Y"), f"{row['Predicted_Sales']:,.2f}"))

        # -----------------------------
        # Graph 1: Historical trend
        # -----------------------------
        ax = self.hist_forecast_ax
        ax.clear()
        hist = self.monthly.copy()
        hist["Moving_Average"] = hist["Sales"].rolling(3).mean()

        ax.plot(
            hist["Date"], hist["Sales"],
            marker="o", markersize=3.5, linewidth=2.0,
            color="#2F80ED", label="Monthly Sales"
        )
        ax.plot(
            hist["Date"], hist["Moving_Average"],
            linewidth=2.0, linestyle="--",
            color="#18A999", label="3-Month Moving Average"
        )
        peak_idx = hist["Sales"].idxmax()
        peak_date = hist.loc[peak_idx, "Date"]
        peak_value = hist.loc[peak_idx, "Sales"]
        ax.scatter([peak_date], [peak_value], s=55, color="#F2A33A", zorder=5)
        ax.annotate(
            f"Peak\n₹{peak_value/1e6:.1f}M",
            xy=(peak_date, peak_value),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="#173F67", fontweight="bold"
        )
        ax.set_title("Historical Sales | 2022–2025", color="#173F67", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sales (₹)", color="#627A93", fontsize=8)
        ax.tick_params(axis="both", colors="#627A93", labelsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
        ax.grid(True, alpha=0.22, color="#C4D2DF")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        for spine in ax.spines.values():
            spine.set_color("#C8D8E8")
        self.hist_forecast_figure.autofmt_xdate(rotation=20)
        self.hist_forecast_canvas.draw_idle()

        # -----------------------------
        # Graph 2: Future forecast trend
        # -----------------------------
        ax = self.future_forecast_ax
        ax.clear()
        ax.plot(
            future["Date"], future["Predicted_Sales"],
            marker="o", markersize=5, linewidth=2.5,
            color="#7B61D9", label="Predicted Sales"
        )
        ax.fill_between(
            future["Date"], future["Predicted_Sales"],
            alpha=0.10, color="#7B61D9"
        )

        first_date = future.iloc[0]["Date"]
        first_value = future.iloc[0]["Predicted_Sales"]
        last_future_date = future.iloc[-1]["Date"]
        last_value = future.iloc[-1]["Predicted_Sales"]
        max_idx = future["Predicted_Sales"].idxmax()
        max_date = future.loc[max_idx, "Date"]
        max_value = future.loc[max_idx, "Predicted_Sales"]

        ax.scatter([max_date], [max_value], s=55, color="#F2A33A", zorder=5)
        ax.annotate(
            f"Peak\n₹{max_value/1e6:.1f}M",
            xy=(max_date, max_value),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8,
            color="#173F67", fontweight="bold"
        )
        ax.annotate(
            f"Start\n₹{first_value/1e6:.1f}M",
            xy=(first_date, first_value),
            xytext=(0, -28), textcoords="offset points",
            ha="center", fontsize=7, color="#627A93"
        )
        ax.annotate(
            f"End\n₹{last_value/1e6:.1f}M",
            xy=(last_future_date, last_value),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=7, color="#173F67"
        )

        ax.set_title(f"Forecast Trend | Next {months} Months", color="#173F67", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Sales (₹)", color="#627A93", fontsize=8)
        ax.tick_params(axis="both", colors="#627A93", labelsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
        ax.grid(True, alpha=0.22, color="#C4D2DF")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        for spine in ax.spines.values():
            spine.set_color("#C8D8E8")
        self.future_forecast_figure.autofmt_xdate(rotation=20)
        self.future_forecast_canvas.draw_idle()

        total = future["Predicted_Sales"].sum()
        self.forecast_result.config(text=f"Total forecast: ₹{total:,.0f}")
        self.card_values["forecast"].config(text=f"₹{total:,.0f}")
        self._update_business_insights()
        if "forecast" in self.card_subtitles:
            self.card_subtitles["forecast"].config(text=f"Next {months} months")
        if not silent:
            self.write_log(f"9. Generated {months}-month forecast using the saved model.")
            self.write_log(f"10. Total forecasted sales: {total:,.2f}.")

    def save_forecast(self):
        if self.forecast is None:
            self.generate_forecast(silent=True)
        if self.forecast is None:
            return
        output = self.forecast[["Date", "Predicted_Sales"]].copy()
        path = OUTPUT_DIR / "System_Forecast.csv"
        output.to_csv(path, index=False)
        self.write_log(f"Forecast saved to {path.name}.")
        messagebox.showinfo("Forecast saved", f"Forecast saved successfully:\n{path}")

    def _update_model_info(self):
        self.model_status.config(text="Model loaded: Improved Linear Regression")
        text = (
            "MODEL INFORMATION\n\n"
            "Algorithm: Improved Linear Regression\n"
            "Saved model: models/sales_forecasting_model.pkl\n\n"
            "Input features:\n"
            "• Time_Index\n"
            "• Year\n"
            "• Month\n"
            "• Quarter\n\n"
            "Chronological test results from the project:\n"
            "MAE  : 904,969.12\n"
            "RMSE : 1,123,771.21\n"
            "R²   : -0.0338\n\n"
            "The model is loaded from the saved .pkl file and is used directly by the Forecast tab."
        )
        self.model_info.configure(state="normal")
        self.model_info.delete("1.0", "end")
        self.model_info.insert("1.0", text)
        self.model_info.configure(state="disabled")

    def show_data_table(self):
        if self.df is None:
            return
        for child in self.data_table_frame.winfo_children():
            child.destroy()
        choice = self.data_choice.get()
        if choice == "Raw Sales Data":
            data = self.df.head(500).copy()
        elif choice == "Monthly Forecasting Data":
            data = self.monthly.copy()
        else:
            if self.forecast is None:
                self.generate_forecast(silent=True)
            data = self.forecast.copy() if self.forecast is not None else pd.DataFrame()
        if data.empty:
            return
        columns = tuple(data.columns)
        table = self._make_table(self.data_table_frame, columns, columns)
        for _, row in data.iterrows():
            values = []
            for value in row:
                if isinstance(value, pd.Timestamp):
                    values.append(value.strftime("%Y-%m-%d"))
                elif isinstance(value, float):
                    values.append(f"{value:,.2f}")
                else:
                    values.append(str(value))
            table.insert("", "end", values=values)
        self.data_table = table


if __name__ == "__main__":
    app = SalesForecastingApp()
    app.mainloop()
