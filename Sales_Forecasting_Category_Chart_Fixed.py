from pathlib import Path
import sys
import re
import ctypes

# Keep Tkinter from being enlarged twice on high-DPI Windows displays.
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

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
        self.active_data_name = DATA_FILE.name
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
        self._install_chart_hover_tooltips()

    def _install_chart_hover_tooltips(self):
        """Add readable hover details to all embedded Matplotlib charts."""
        canvas_names = (
            "dashboard_canvas", "analysis_canvas", "business_analysis_canvas",
            "hist_forecast_canvas", "future_forecast_canvas"
        )
        self._hover_annotations = []
        self._hover_canvases = []
        for name in canvas_names:
            canvas = getattr(self, name, None)
            if canvas is None:
                continue
            self._hover_canvases.append(canvas)
            for ax in canvas.figure.axes:
                annotation = ax.annotate(
                    "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#2F80ED", alpha=0.97),
                    fontsize=10, color="#173F67", zorder=100
                )
                annotation.set_visible(False)
                self._hover_annotations.append((ax, annotation))
            canvas.mpl_connect("motion_notify_event", self._on_chart_hover)

    def _refresh_business_hover_annotation(self):
        """Reattach hover annotation after the business chart switches plot types."""
        ax = getattr(self, "business_analysis_ax", None)
        if ax is None:
            return
        if not any(existing_ax is ax for existing_ax, _ in self._hover_annotations):
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#2F80ED", alpha=0.97),
                fontsize=10, color="#173F67", zorder=100
            )
            annotation.set_visible(False)
            self._hover_annotations.append((ax, annotation))

    def _on_chart_hover(self, event):
        """Show the nearest plotted value while the cursor moves over a chart."""
        for ax, annotation in self._hover_annotations:
            if event.inaxes is not ax:
                annotation.set_visible(False)
        ax = event.inaxes
        if ax is None or event.x is None or event.y is None:
            for canvas in self._hover_canvases:
                canvas.draw_idle()
            return
        target = None
        best_distance = 13.0
        for line in ax.lines:
            try:
                xs, ys = line.get_data()
                if len(xs) == 0:
                    continue
                points = ax.transData.transform(list(zip(xs, ys)))
                import numpy as _np
                distances = _np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
                idx = int(distances.argmin())
                if distances[idx] < best_distance:
                    best_distance = float(distances[idx])
                    target = (xs[idx], ys[idx], line.get_label())
            except Exception:
                pass
        for collection in ax.collections:
            try:
                offsets = collection.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue
                points = ax.transData.transform(offsets)
                import numpy as _np
                distances = _np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
                idx = int(distances.argmin())
                if distances[idx] < best_distance:
                    best_distance = float(distances[idx])
                    target = (offsets[idx][0], offsets[idx][1], "Data point")
            except Exception:
                pass
        if target is None:
            for patch in ax.patches:
                try:
                    contains, _ = patch.contains(event)
                    if contains:
                        label = patch.get_label()
                        if not label or label.startswith("_"):
                            label = "Selected segment"
                        target = (None, None, label)
                        break
                except Exception:
                    pass
        for current_ax, annotation in self._hover_annotations:
            if current_ax is ax:
                if target is None:
                    annotation.set_visible(False)
                else:
                    x, y, label = target
                    if x is None:
                        annotation.xy = (event.xdata, event.ydata)
                        annotation.set_text(str(label))
                    else:
                        annotation.xy = (x, y)
                        try:
                            import matplotlib.dates as _mdates
                            if isinstance(x, (float, int)) and ax.xaxis_date():
                                x_text = _mdates.num2date(float(x)).strftime("%b %Y")
                            else:
                                x_text = f"{x}"
                        except Exception:
                            x_text = f"{x}"
                        try:
                            y_text = f"₹{float(y):,.2f}"
                        except Exception:
                            y_text = f"{y}"
                        annotation.set_text(f"{label} | X: {x_text} | Value: {y_text}")
                    annotation.set_visible(True)
        for canvas in self._hover_canvases:
            canvas.draw_idle()

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
        )
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

        actions = tk.Frame(header, bg="#FFFFFF")
        actions.pack(side="right", padx=20, pady=25)
        ttk.Button(
            actions, text="Upload Sales Data", style="Accent.TButton",
            command=self.upload_sales_data
        ).pack(anchor="e", pady=(0, 8))
        ttk.Button(
            actions, text="Export Active Data",
            style="Secondary.TButton", command=self.export_active_data
        ).pack(anchor="e")
        tk.Label(
            header, text="ANALYZE  •  PREDICT  •  GROW",
            bg="#FFFFFF", fg="#18A999", font=("Segoe UI", 11, "bold"),
        ).pack(side="right", padx=12, pady=30)


    def upload_sales_data(self):
        """Import, validate, preview and activate a business user's sales dataset."""
        path = filedialog.askopenfilename(
            title="Upload Business Sales Data",
            filetypes=[
                ("Sales data files", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
            ],
        )
        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                try:
                    uploaded = pd.read_csv(path)
                except UnicodeDecodeError:
                    uploaded = pd.read_csv(path, encoding="latin-1")
            else:
                uploaded = pd.read_excel(path)

            if uploaded.empty:
                raise ValueError("The selected file is empty.")
            uploaded.columns = [str(c).strip() for c in uploaded.columns]
            original_rows = len(uploaded)
            duplicate_rows = int(uploaded.duplicated().sum())
            missing_before = int(uploaded.isna().sum().sum())

            # Normalize headers: case-insensitive, spaces/hyphens become underscores.
            normalized = {}
            for col in uploaded.columns:
                key = re.sub(r"[\s\-]+", "_", str(col).strip().lower())
                normalized[key] = col

            aliases = {
                "Date": ["date", "order_date", "transaction_date", "invoice_date", "month", "sales_date"],
                "Sales": ["sales", "revenue", "total_sales", "amount", "sales_amount", "total_revenue"],
                "Product": ["product", "product_name", "item", "item_name", "product_id"],
                "Category": ["category", "product_category", "product_type"],
                "Region": ["region", "state", "location", "city", "territory"],
                "Sales_Channel": ["sales_channel", "channel", "sales_type", "channel_type"],
                "Profit": ["profit", "net_profit", "gross_profit", "profit_amount"],
                "Quantity": ["quantity", "qty", "units_sold", "units", "items_sold"],
                "Cost": ["cost", "total_cost", "cost_amount", "expenses"],
            }
            rename = {}
            for standard, candidates in aliases.items():
                for candidate in candidates:
                    if candidate in normalized:
                        rename[normalized[candidate]] = standard
                        break
            uploaded = uploaded.rename(columns=rename)

            missing_required = [c for c in ("Date", "Sales") if c not in uploaded.columns]
            if missing_required:
                raise ValueError(
                    "Required columns are missing: " + ", ".join(missing_required) +
                    ". Your file must have a date column (e.g. Date or Order_Date) "
                    "and a sales amount column (e.g. Sales, Revenue or Amount)."
                )

            # Handle currency symbols/commas in common exported spreadsheets.
            uploaded["Date"] = pd.to_datetime(uploaded["Date"], errors="coerce", dayfirst=True)
            for col in ("Sales", "Profit", "Quantity", "Cost"):
                if col in uploaded.columns:
                    cleaned = uploaded[col].astype(str).str.replace(",", "", regex=False)
                    cleaned = cleaned.str.replace("₹", "", regex=False).str.replace("$", "", regex=False)
                    cleaned = cleaned.str.replace("£", "", regex=False).str.strip()
                    uploaded[col] = pd.to_numeric(cleaned, errors="coerce")

            invalid_date_rows = int(uploaded["Date"].isna().sum())
            invalid_sales_rows = int(uploaded["Sales"].isna().sum())
            uploaded = uploaded.dropna(subset=["Date", "Sales"]).copy()
            if uploaded.empty:
                raise ValueError("No valid rows remain. Check the date and sales values.")
            if (uploaded["Sales"] < 0).any():
                raise ValueError(
                    "Negative sales values were found. Please correct returns/refunds or "
                    "confirm that your file uses non-negative sales totals before uploading."
                )

            # Keep optional dimensions usable even when another business uses different columns.
            defaults = {
                "Product": "Unspecified product",
                "Category": "Unspecified category",
                "Region": "Unspecified region",
                "Sales_Channel": "Unspecified channel",
            }
            for col, default in defaults.items():
                if col not in uploaded.columns:
                    uploaded[col] = default
                uploaded[col] = uploaded[col].fillna(default).astype(str).str.strip()
                uploaded.loc[uploaded[col].eq(""), col] = default

            if "Profit" not in uploaded.columns:
                uploaded["Profit"] = 0.0
            uploaded["Profit"] = pd.to_numeric(uploaded["Profit"], errors="coerce").fillna(0.0)
            if "Quantity" not in uploaded.columns:
                uploaded["Quantity"] = 1
            uploaded["Quantity"] = pd.to_numeric(uploaded["Quantity"], errors="coerce").fillna(1)
            if "Cost" not in uploaded.columns:
                uploaded["Cost"] = uploaded["Sales"] - uploaded["Profit"]
            uploaded["Cost"] = pd.to_numeric(uploaded["Cost"], errors="coerce").fillna(0.0)

            uploaded["Year"] = uploaded["Date"].dt.year
            uploaded["Month"] = uploaded["Date"].dt.month
            uploaded["Month_Name"] = uploaded["Date"].dt.month_name()

            # Aggregate transactions into monthly sales. Require at least 3 periods for a
            # basic trend model; this is a simple baseline, not a guarantee of accuracy.
            monthly = uploaded.groupby(uploaded["Date"].dt.to_period("M"))["Sales"].sum().reset_index()
            monthly["Date"] = monthly["Date"].dt.to_timestamp()
            monthly = monthly.sort_values("Date").reset_index(drop=True)
            if len(monthly) < 3:
                raise ValueError(
                    f"Only {len(monthly)} distinct month(s) of valid sales data were found. "
                    "Please upload at least 3 months; 12 or more months is recommended."
                )

            monthly["Time_Index"] = np.arange(len(monthly))
            monthly["Year"] = monthly["Date"].dt.year
            monthly["Month"] = monthly["Date"].dt.month
            monthly["Quarter"] = monthly["Date"].dt.quarter

            from sklearn.linear_model import LinearRegression
            features = ["Time_Index", "Year", "Month", "Quarter"]
            new_model = LinearRegression()
            new_model.fit(monthly[features], monthly["Sales"])

            # Only activate the new data after validation and model fitting both succeed.
            self.df = uploaded.reset_index(drop=True)
            self.monthly = monthly
            self.model = new_model
            self.forecast = None
            self.active_data_name = Path(path).name

            # Record useful validation details in the application's learning log.
            kept_rows = len(uploaded)
            dropped_rows = original_rows - kept_rows
            self.write_log(f"Uploaded dataset: {self.active_data_name}")
            self.write_log(
                f"Validation: {kept_rows:,} usable rows; {len(monthly)} distinct months; "
                f"{duplicate_rows:,} duplicate rows detected."
            )
            self.write_log(
                f"Data cleaning: {invalid_date_rows:,} invalid dates and "
                f"{invalid_sales_rows:,} invalid sales values detected before filtering."
            )
            self.write_log("Trained an in-memory Linear Regression baseline on the uploaded dataset.")

            # Refresh all views to prevent stale KPIs/graphs from the previous dataset.
            self._update_dashboard()
            self.show_analysis()
            self.show_business_analysis()
            self._refresh_product_suggestions()
            self.show_data_table()
            self.generate_forecast(silent=True)
            self._show_page("Dashboard")

            summary = (
                f"Successfully loaded: {Path(path).name}\n\n"
                f"Rows in file: {original_rows:,}\n"
                f"Usable rows: {kept_rows:,}\n"
                f"Rows excluded for invalid Date/Sales: {dropped_rows:,}\n"
                f"Duplicate rows detected: {duplicate_rows:,}\n"
                f"Missing cells before cleaning: {missing_before:,}\n"
                f"Monthly periods: {len(monthly)}\n"
                f"Date range: {uploaded['Date'].min():%d %b %Y} – {uploaded['Date'].max():%d %b %Y}\n\n"
                "The dashboard and graphs now use this uploaded dataset.\n"
                "The original saved model file was not overwritten."
            )
            messagebox.showinfo("Sales Data Loaded", summary)

        except Exception as exc:
            messagebox.showerror(
                "Could Not Upload Data",
                f"The file could not be activated.\n\n{exc}\n\n"
                "Your currently loaded dataset has been kept unchanged."
            )

    def export_active_data(self):
        """Save the currently active dataset to CSV without modifying the source file."""
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Active Sales Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="Active_Sales_Data.csv",
        )
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            messagebox.showinfo("Export Complete", f"Active data exported to:\n{path}")
            self.write_log(f"Exported active dataset to {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))

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

        # Model management remains internal for forecasting, but is not shown as a separate page.
        pages = [
            ("Dashboard", "⌂"), ("Sales Analysis", "▥"), ("Business Analysis", "◆"), ("Forecast", "↗"),
            ("Business Insights", "★"), ("Product Suggestions", "✦"), ("Data", "▤"), ("Learning Log", "▣")
        ]
        self.page_buttons = {}
        self.pages = {}

        self.dashboard_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.analysis_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.business_analysis_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.forecast_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.insights_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.product_suggestions_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.model_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.data_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        self.log_tab = tk.Frame(self.page_host, bg="#F5F8FC")
        frames = [self.dashboard_tab, self.analysis_tab, self.business_analysis_tab, self.forecast_tab, self.insights_tab, self.product_suggestions_tab, self.data_tab, self.log_tab]

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
        self._build_product_suggestions()
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
            figsize=(11.5, 5.2), dpi=90,
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
                "-font", "Segoe UI 13",
                "-activestyle", "none"
            )
        except tk.TclError:
            pass

    def show_business_analysis(self):
        # Startup/upload callbacks can occasionally fire before this page is built.
        if self.df is None or not hasattr(self, "business_choice"):
            return
        if not hasattr(self, "business_kpi_values") or not hasattr(self, "business_analysis_figure"):
            return

        choice = self.business_choice.get()
        if choice not in ("Top-Selling Products", "Leading Categories", "Leading Regions", "Leading Sales Channels"):
            choice = "Top-Selling Products"

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

        # Always use a normal Cartesian axis. This prevents polar-axis
        # switching from corrupting the layout when changing selections.
        self.business_analysis_figure.clear()
        ax = self.business_analysis_figure.add_subplot(111)
        ax.set_facecolor("#FFFFFF")
        self.business_analysis_ax = ax

        labels = [str(x) for x in series.index]
        values = series.values.astype(float)

        # 1. PRODUCTS: original horizontal lollipop ranking from the reference UI.
        if choice == "Top-Selling Products":
            ordered = series.sort_values(ascending=True)
            y = np.arange(len(ordered))
            max_value = float(ordered.max()) if len(ordered) else 1.0
            ax.hlines(y, 0, ordered.values, color="#C7D9EC", linewidth=3)
            ax.scatter(
                ordered.values, y, s=90, color="#2F80ED",
                edgecolor="white", linewidth=1.2, zorder=3
            )
            for yi, value in zip(y, ordered.values):
                ax.annotate(
                    f"₹{value/1e6:.1f}M", (value, yi),
                    xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color="#173F67"
                )
            ax.set_yticks(y)
            ax.set_yticklabels([str(x) for x in ordered.index], fontsize=9, color="#173F67")
            ax.set_xlabel("Sales contribution (₹)", color="#627A93", fontsize=10)
            ax.set_title("Top-Selling Products", color="#173F67", fontsize=15, fontweight="bold", pad=10)
            ax.grid(axis="x", alpha=0.18, color="#C4D2DF")
            ax.set_xlim(0, max_value * 1.18)
            ax.tick_params(axis="x", labelsize=8, colors="#627A93")
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        # 2. CATEGORIES: original donut with category-share legend.
        elif choice == "Leading Categories":
            # Keep the donut on the left and put category details in a separate,
            # readable legend area on the right (prevents labels overlapping slices).
            positive = [(name, float(value)) for name, value in zip(labels, values)
                        if np.isfinite(value) and float(value) > 0]
            cat_labels = [item[0] for item in positive]
            cat_values = np.array([item[1] for item in positive], dtype=float)
            cat_total = float(cat_values.sum()) if len(cat_values) else 0.0
            ax.set_position([0.06, 0.10, 0.42, 0.78])
            if cat_total <= 0:
                ax.text(0.5, 0.5, "No category sales data available", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="#627A93")
            else:
                wedges, _ = ax.pie(
                    cat_values, startangle=90, counterclock=False,
                    colors=["#2878B5", "#FF851B", "#2CA02C", "#D62728", "#9467BD", "#17A2A4"],
                    wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
                    radius=0.90
                )
                leader = cat_labels[int(np.argmax(cat_values))]
                ax.text(0, 0.09, "Leading Category", ha="center", va="center",
                        fontsize=9, color="#627A93")
                ax.text(0, -0.10, leader, ha="center", va="center", fontsize=11,
                        fontweight="bold", color="#173F67", wrap=True)
                legend_labels = [
                    f"{name}   |   {value/cat_total*100:.1f}%   |   ₹{value/1e6:.1f}M"
                    for name, value in zip(cat_labels, cat_values)
                ]
                ax.legend(
                    wedges, legend_labels, title="Category  |  Share  |  Sales",
                    title_fontsize=10, fontsize=9, loc="center left",
                    bbox_to_anchor=(1.10, 0.50), borderaxespad=0,
                    labelspacing=1.5, handlelength=1.2, handletextpad=0.8,
                    frameon=False
                )
                ax.set_aspect("equal")
            ax.set_title("Sales Share by Category", color="#173F67", fontsize=15,
                         fontweight="bold", pad=10)

        # 3. REGIONS: original teal line/area comparison (not a lollipop chart).
        elif choice == "Leading Regions":
            x = np.arange(len(labels))
            ax.plot(x, values, color="#18A999", linewidth=2.5, marker="o", markersize=8,
                    markerfacecolor="#18A999", markeredgecolor="white", markeredgewidth=1.2)
            ax.fill_between(x, values, alpha=0.12, color="#18A999")
            for xi, value in zip(x, values):
                ax.annotate(f"₹{value/1e6:.1f}M", (xi, value), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=9,
                            fontweight="bold", color="#173F67")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9, color="#173F67")
            ax.set_ylabel("Regional Sales (₹)", color="#627A93", fontsize=9)
            ax.set_title("Regional Sales Comparison", color="#173F67", fontsize=15,
                         fontweight="bold", pad=10)
            ax.grid(axis="y", alpha=0.18, color="#C4D2DF")
            ax.tick_params(axis="y", labelsize=8, colors="#627A93")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
            ax.margins(x=0.05, y=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 4. CHANNELS: original radar chart from the reference UI.
        else:
            ax.remove()
            ax = self.business_analysis_figure.add_subplot(111, projection="polar")
            ax.set_facecolor("#FFFFFF")
            self.business_analysis_ax = ax
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            radar_values = values.tolist()
            angles += angles[:1]
            radar_values += radar_values[:1]
            ax.plot(angles, radar_values, color="#7B61D9", linewidth=2.5)
            ax.fill(angles, radar_values, color="#7B61D9", alpha=0.16)
            ax.scatter(angles[:-1], radar_values[:-1], color="#7B61D9", s=50,
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=9, color="#173F67", fontweight="bold")
            ax.set_title("Sales Channel Comparison", color="#173F67", fontsize=15,
                         fontweight="bold", pad=18)
            ax.tick_params(axis="y", labelsize=7, colors="#627A93")
            ax.grid(alpha=0.18, color="#C4D2DF")
            for angle, value in zip(angles[:-1], values):
                ax.annotate(f"₹{value/1e6:.1f}M", (angle, value), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=8,
                            fontweight="bold", color="#173F67")

        for spine in ax.spines.values():
            spine.set_color("#C8D8E8")

        if choice == "Leading Categories":
            self.business_analysis_figure.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.10)
        else:
            self.business_analysis_figure.subplots_adjust(
                left=0.08, right=0.95, top=0.88, bottom=0.12
            )
        self.business_analysis_canvas.draw_idle()
        self._refresh_business_hover_annotation()
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
        self.insight_summary.pack(fill="x", pady=(0, 8))
        for i in range(4):
            self.insight_summary.columnconfigure(i, weight=1)

        tk.Label(self.insights_tab, text="Business Insights", bg="#F5F8FC",
                 fg="#173F67", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(self.insights_tab,
                 text="Historical performance and 12-month forecast at a glance",
                 bg="#F5F8FC", fg="#6C839A", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 8))

        self.business_card_values = {}
        cards = [
            ("forecast_total", "Forecast Total", "#2F80ED"),
            ("forecast_avg", "Monthly Average", "#18A999"),
            ("forecast_growth", "First → Last Growth", "#7B61D9"),
            ("peak_forecast", "Peak Forecast", "#F2A33A")
        ]
        for i, (key, title, accent) in enumerate(cards):
            card = tk.Frame(self.insight_summary, bg="#FFFFFF",
                            highlightbackground="#C8D8E8", highlightthickness=1)
            card.grid(row=0, column=i, padx=4, sticky="nsew")
            tk.Frame(card, bg=accent, height=4).pack(fill="x")
            body = tk.Frame(card, bg="#FFFFFF")
            body.pack(fill="both", expand=True, padx=10, pady=7)
            tk.Label(body, text=title.upper(), bg="#FFFFFF", fg="#627A93",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            value = tk.Label(body, text="--", bg="#FFFFFF", fg="#173F67",
                             font=("Segoe UI", 13, "bold"))
            value.pack(anchor="w", pady=(3, 0))
            self.business_card_values[key] = value

        columns = tk.Frame(self.insights_tab, bg="#F5F8FC")
        columns.pack(fill="both", expand=True)
        columns.columnconfigure(0, weight=1)
        columns.columnconfigure(1, weight=1)
        columns.rowconfigure(0, weight=1)

        self.performance_frame = self._insight_chart_card(
            columns, 0, "Historical Sales by Year")
        self.forecast_frame = self._insight_chart_card(
            columns, 1, "12-Month Sales Forecast")

        self.performance_text = tk.Text(
            self.performance_frame, wrap="word", state="disabled",
            bg="#F8FBFE", fg="#35546F", relief="flat",
            font=("Segoe UI", 9), padx=9, pady=6, height=5)
        self.performance_text.pack(fill="x", padx=8, pady=(0, 8))
        self.forecast_story = tk.Text(
            self.forecast_frame, wrap="word", state="disabled",
            bg="#F8FBFE", fg="#35546F", relief="flat",
            font=("Segoe UI", 9), padx=9, pady=6, height=5)
        self.forecast_story.pack(fill="x", padx=8, pady=(0, 8))

        self.insight_history_figure = Figure(figsize=(5.0, 3.0), dpi=85, facecolor="#FFFFFF")
        self.insight_history_ax = self.insight_history_figure.add_subplot(111)
        self.insight_history_canvas = FigureCanvasTkAgg(self.insight_history_figure, master=self.performance_frame)
        self.insight_history_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.insight_history_tooltip = self.insight_history_ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#2F80ED"),
            fontsize=8, visible=False)
        self.insight_history_canvas.mpl_connect("motion_notify_event", self._on_history_hover)

        self.insight_forecast_figure = Figure(figsize=(5.0, 3.0), dpi=85, facecolor="#FFFFFF")
        self.insight_forecast_ax = self.insight_forecast_figure.add_subplot(111)
        self.insight_forecast_canvas = FigureCanvasTkAgg(self.insight_forecast_figure, master=self.forecast_frame)
        self.insight_forecast_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.insight_forecast_tooltip = self.insight_forecast_ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#18A999"),
            fontsize=8, visible=False)
        self.insight_forecast_canvas.mpl_connect("motion_notify_event", self._on_forecast_hover)

    def _insight_chart_card(self, parent, column, title):
        frame = tk.Frame(parent, bg="#FFFFFF", highlightbackground="#C8D8E8", highlightthickness=1)
        frame.grid(row=0, column=column, sticky="nsew", padx=4)
        tk.Label(frame, text=title, bg="#FFFFFF", fg="#173F67",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(9, 3))
        return frame

    def _on_history_hover(self, event):
        if event.inaxes != self.insight_history_ax or not hasattr(self, "insight_history_bars"):
            if self.insight_history_tooltip.get_visible():
                self.insight_history_tooltip.set_visible(False)
                self.insight_history_canvas.draw_idle()
            return
        for bar, year_label, sales_value in self.insight_history_bars:
            if bar.contains(event)[0]:
                self.insight_history_tooltip.xy = (bar.get_x() + bar.get_width()/2, bar.get_height())
                self.insight_history_tooltip.set_text(f"{year_label}: ₹{sales_value:,.0f}")
                self.insight_history_tooltip.set_visible(True)
                self.insight_history_canvas.draw_idle()
                return
        if self.insight_history_tooltip.get_visible():
            self.insight_history_tooltip.set_visible(False)
            self.insight_history_canvas.draw_idle()

    def _on_forecast_hover(self, event):
        if event.inaxes != self.insight_forecast_ax or event.xdata is None or event.ydata is None:
            if self.insight_forecast_tooltip.get_visible():
                self.insight_forecast_tooltip.set_visible(False)
                self.insight_forecast_canvas.draw_idle()
            return
        points = getattr(self, "insight_forecast_points", [])
        if not points:
            return
        index = min(range(len(points)), key=lambda i: abs(points[i][0] - event.xdata))
        xval, yval, month_label, sales_value = points[index]
        yspan = max(0.01, self.insight_forecast_ax.get_ylim()[1] - self.insight_forecast_ax.get_ylim()[0])
        if abs(yval - event.ydata) <= yspan * 0.15:
            self.insight_forecast_tooltip.xy = (xval, yval)
            self.insight_forecast_tooltip.set_text(f"{month_label}: ₹{sales_value:,.0f}")
            self.insight_forecast_tooltip.set_visible(True)
            self.insight_forecast_canvas.draw_idle()
        elif self.insight_forecast_tooltip.get_visible():
            self.insight_forecast_tooltip.set_visible(False)
            self.insight_forecast_canvas.draw_idle()

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
        # Historical sales chart with hover details
        hax = self.insight_history_ax
        hax.clear()
        years = self.df.groupby("Year")["Sales"].sum()
        labels = [str(y) for y in years.index]
        millions = years.values / 1_000_000
        bars = hax.bar(labels, millions, color="#2F80ED", edgecolor="white")
        self.insight_history_bars = [(bar, str(y), float(v))
                                     for bar, y, v in zip(bars, years.index, years.values)]
        hax.set_ylabel("Sales (₹ million)", fontsize=8)
        hax.tick_params(axis="both", labelsize=8)
        hax.grid(axis="y", alpha=0.22)
        hax.set_axisbelow(True)
        hax.spines["top"].set_visible(False)
        hax.spines["right"].set_visible(False)
        for bar, value in zip(bars, millions):
            hax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        self.insight_history_tooltip.set_visible(False)
        self.insight_history_figure.tight_layout()
        self.insight_history_canvas.draw_idle()

        # Forecast chart with hover details
        fax = self.insight_forecast_ax
        fax.clear()
        forecast_dates = self.forecast["Date"].tolist()
        raw_values = self.forecast["Predicted_Sales"].astype(float).tolist()
        xvals = list(range(len(raw_values)))
        yvals = [v / 1_000_000 for v in raw_values]
        fax.plot(xvals, yvals, marker="o", markersize=3, linewidth=2, color="#18A999")
        fax.fill_between(xvals, yvals, alpha=0.12, color="#18A999")
        self.insight_forecast_points = [
            (float(i), float(yvals[i]), forecast_dates[i].strftime("%b %Y"), raw_values[i])
            for i in range(len(raw_values))]
        fax.set_xticks(xvals[::2])
        fax.set_xticklabels([forecast_dates[i].strftime("%b %y") for i in xvals[::2]],
                            rotation=25, ha="right")
        fax.set_ylabel("Forecast (₹ million)", fontsize=8)
        fax.tick_params(axis="both", labelsize=7)
        fax.grid(axis="y", alpha=0.22)
        fax.set_axisbelow(True)
        fax.spines["top"].set_visible(False)
        fax.spines["right"].set_visible(False)
        self.insight_forecast_tooltip.set_visible(False)
        self.insight_forecast_figure.tight_layout()
        self.insight_forecast_canvas.draw_idle()

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


    def _build_product_suggestions(self):
        """Create a data-driven product suggestions page using the active dataset."""
        top = tk.Frame(self.product_suggestions_tab, bg="#F5F8FC")
        top.pack(fill="x", pady=(0, 10))
        tk.Label(top, text="Product Suggestions", bg="#F5F8FC", fg="#173F67",
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(top,
                 text="Explore best sellers and identify products that may deserve a careful promotion review.",
                 bg="#F5F8FC", fg="#6C839A", font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 0))

        filters = tk.Frame(self.product_suggestions_tab, bg="#FFFFFF",
                           highlightbackground="#C8D8E8", highlightthickness=1)
        filters.pack(fill="x", pady=(0, 10))
        tk.Label(filters, text="Region", bg="#FFFFFF", fg="#173F67",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, padx=(14, 6), pady=12, sticky="w")
        self.suggestion_region = ttk.Combobox(filters, state="readonly", width=18)
        self.suggestion_region.grid(row=0, column=1, padx=(0, 16), pady=12)
        self.suggestion_region.bind("<<ComboboxSelected>>", lambda _e: self._refresh_product_suggestions())

        tk.Label(filters, text="Category", bg="#FFFFFF", fg="#173F67",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=(0, 6), pady=12, sticky="w")
        self.suggestion_category = ttk.Combobox(filters, state="readonly", width=20)
        self.suggestion_category.grid(row=0, column=3, padx=(0, 16), pady=12)
        self.suggestion_category.bind("<<ComboboxSelected>>", lambda _e: self._refresh_product_suggestions())

        tk.Button(filters, text="Refresh Suggestions", command=self._refresh_product_suggestions,
                  bg="#0B63D9", fg="white", activebackground="#084DA8",
                  relief="flat", padx=14, pady=7, font=("Segoe UI", 10, "bold")
                  ).grid(row=0, column=4, padx=8, pady=8)

        cards = tk.Frame(self.product_suggestions_tab, bg="#F5F8FC")
        cards.pack(fill="x", pady=(0, 10))
        self.suggestion_summary_labels = {}
        summary_defs = [
            ("top", "BEST-SELLING PRODUCT", "—", "#2F80ED"),
            ("region", "REGION FILTER", "All regions", "#18A999"),
            ("promotion", "PROMOTION REVIEW", "—", "#7B61D9"),
        ]
        for col, (key, title, value, accent) in enumerate(summary_defs):
            cards.columnconfigure(col, weight=1)
            card = tk.Frame(cards, bg="#FFFFFF", highlightbackground="#C8D8E8", highlightthickness=1)
            card.grid(row=0, column=col, padx=5, sticky="nsew")
            tk.Frame(card, bg=accent, height=4).pack(fill="x")
            tk.Label(card, text=title, bg="#FFFFFF", fg="#627A93",
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(9, 2))
            label = tk.Label(card, text=value, bg="#FFFFFF", fg="#173F67",
                             font=("Segoe UI", 14, "bold"), wraplength=260, justify="left")
            label.pack(anchor="w", padx=12, pady=(0, 10))
            self.suggestion_summary_labels[key] = label

        body = tk.Frame(self.product_suggestions_tab, bg="#F5F8FC")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        tk.Label(body, text="Top-selling products", bg="#F5F8FC", fg="#173F67",
                 font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(2, 6))
        tk.Label(body, text="Products to review for a possible promotion", bg="#F5F8FC", fg="#173F67",
                 font=("Segoe UI", 14, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(2, 6))

        left = tk.Frame(body, bg="#FFFFFF", highlightbackground="#C8D8E8", highlightthickness=1)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        right = tk.Frame(body, bg="#FFFFFF", highlightbackground="#C8D8E8", highlightthickness=1)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        cols = ("product", "sales", "quantity")
        self.top_products_tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        for col, label, width in [("product", "Product", 180), ("sales", "Total sales (₹)", 120), ("quantity", "Quantity", 90)]:
            self.top_products_tree.heading(col, text=label)
            self.top_products_tree.column(col, width=width, anchor="w" if col == "product" else "e")
        self.top_products_tree.pack(fill="both", expand=True, padx=8, pady=8)

        pcols = ("product", "sales", "profit", "margin")
        self.promotion_tree = ttk.Treeview(right, columns=pcols, show="headings", height=12)
        for col, label, width in [("product", "Product", 145), ("sales", "Sales (₹)", 95),
                                  ("profit", "Profit (₹)", 95), ("margin", "Margin", 80)]:
            self.promotion_tree.heading(col, text=label)
            self.promotion_tree.column(col, width=width, anchor="w" if col == "product" else "e")
        self.promotion_tree.pack(fill="both", expand=True, padx=8, pady=8)

        self.suggestion_note = tk.Label(
            self.product_suggestions_tab,
            text="Promotion candidates are based on relatively lower sales and positive recorded profit; they are not proof that a discount will improve results.",
            bg="#F5F8FC", fg="#6C839A", font=("Segoe UI", 9), wraplength=1000, justify="left"
        )
        self.suggestion_note.pack(anchor="w", pady=(8, 0))
        self._refresh_product_suggestions()

    def _refresh_product_suggestions(self):
        """Update the suggestion tables from the currently active dataframe."""
        if not hasattr(self, "top_products_tree") or self.df is None:
            return

        df = self.df.copy()
        if "Product" not in df.columns or "Sales" not in df.columns:
            self.suggestion_note.configure(
                text="This dataset needs Product and Sales columns to show product suggestions. Upload a dataset containing those columns."
            )
            return

        if "Region" not in df.columns:
            df["Region"] = "Unspecified region"
        if "Category" not in df.columns:
            df["Category"] = "Unspecified category"
        if "Quantity" not in df.columns:
            df["Quantity"] = 0
        if "Profit" not in df.columns:
            if "Cost" in df.columns:
                df["Profit"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0) - pd.to_numeric(df["Cost"], errors="coerce").fillna(0)
            else:
                df["Profit"] = 0

        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)

        regions = ["All regions"] + sorted(df["Region"].fillna("Unspecified region").astype(str).unique().tolist())
        categories = ["All categories"] + sorted(df["Category"].fillna("Unspecified category").astype(str).unique().tolist())
        old_region = self.suggestion_region.get() if hasattr(self, "suggestion_region") else "All regions"
        old_category = self.suggestion_category.get() if hasattr(self, "suggestion_category") else "All categories"
        self.suggestion_region["values"] = regions
        self.suggestion_category["values"] = categories
        if old_region not in regions:
            old_region = "All regions"
        if old_category not in categories:
            old_category = "All categories"
        self.suggestion_region.set(old_region)
        self.suggestion_category.set(old_category)

        filtered = df.copy()
        if old_region != "All regions":
            filtered = filtered[filtered["Region"].fillna("Unspecified region").astype(str) == old_region]
        if old_category != "All categories":
            filtered = filtered[filtered["Category"].fillna("Unspecified category").astype(str) == old_category]

        for tree in (self.top_products_tree, self.promotion_tree):
            for item in tree.get_children():
                tree.delete(item)

        if filtered.empty:
            self.suggestion_summary_labels["top"].configure(text="No matching products")
            self.suggestion_summary_labels["region"].configure(text=old_region)
            self.suggestion_summary_labels["promotion"].configure(text="No candidates")
            return

        grouped = filtered.groupby("Product", dropna=False).agg(
            Sales=("Sales", "sum"), Quantity=("Quantity", "sum"), Profit=("Profit", "sum")
        ).reset_index()
        grouped["Product"] = grouped["Product"].fillna("Unspecified product").astype(str)
        grouped["Margin"] = grouped.apply(lambda r: (r["Profit"] / r["Sales"] * 100) if r["Sales"] > 0 else 0, axis=1)

        top = grouped.sort_values("Sales", ascending=False).head(10)
        for _, row in top.iterrows():
            self.top_products_tree.insert("", "end", values=(
                row["Product"], f'{row["Sales"]:,.2f}', f'{row["Quantity"]:,.0f}'
            ))

        # Candidates: below-median sales with positive profit, ranked by profit margin.
        median_sales = grouped["Sales"].median()
        candidates = grouped[(grouped["Sales"] < median_sales) & (grouped["Profit"] > 0)].sort_values(
            ["Margin", "Sales"], ascending=[False, True]
        ).head(10)
        for _, row in candidates.iterrows():
            self.promotion_tree.insert("", "end", values=(
                row["Product"], f'{row["Sales"]:,.2f}', f'{row["Profit"]:,.2f}', f'{row["Margin"]:.1f}%'
            ))

        best = grouped.sort_values("Sales", ascending=False).iloc[0]
        self.suggestion_summary_labels["top"].configure(text=f'{best["Product"]}  |  ₹{best["Sales"]:,.0f}')
        self.suggestion_summary_labels["region"].configure(text=old_region)
        self.suggestion_summary_labels["promotion"].configure(text=f"{len(candidates)} candidate(s)")
        if candidates.empty:
            self.suggestion_note.configure(
                text="No products met the current rule (sales below the filtered median and positive recorded profit). Try another region/category. A discount should always be tested carefully."
            )
        else:
            self.suggestion_note.configure(
                text="Promotion candidates are products with sales below the filtered median and positive recorded profit. This is a screening list only—not a guarantee that discounting will increase sales or profit. Inventory/stock is not included in this dataset."
            )

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
            self._refresh_product_suggestions()
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
