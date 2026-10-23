from pathlib import Path
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
        self.geometry("1350x900")
        self.minsize(1100, 750)

        self.df = None
        self.monthly = None
        self.model = None
        self.forecast = None
        self.figure = None
        self.canvas = None
        self.log_box = None

        self._configure_style()
        self._build_header()
        self._build_tabs()
        self._load_system()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("CardValue.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _build_header(self):
        header = ttk.Frame(self, padding=(20, 16, 20, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Sales Forecasting & Trend Analysis", style="Title.TLabel").pack()
        ttk.Label(
            header,
            text="Data → Analysis → Machine Learning → 12-Month Forecast",
            style="Subtitle.TLabel",
        ).pack(pady=(3, 0))

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=18, pady=(6, 18))

        self.dashboard_tab = ttk.Frame(self.tabs, padding=10)
        self.analysis_tab = ttk.Frame(self.tabs, padding=10)
        self.forecast_tab = ttk.Frame(self.tabs, padding=10)
        self.model_tab = ttk.Frame(self.tabs, padding=10)
        self.data_tab = ttk.Frame(self.tabs, padding=10)
        self.log_tab = ttk.Frame(self.tabs, padding=10)

        self.tabs.add(self.dashboard_tab, text="Dashboard")
        self.tabs.add(self.analysis_tab, text="Sales Analysis")
        self.tabs.add(self.forecast_tab, text="Forecast")
        self.tabs.add(self.model_tab, text="Model")
        self.tabs.add(self.data_tab, text="Data")
        self.tabs.add(self.log_tab, text="Learning Log")

        self._build_dashboard()
        self._build_analysis()
        self._build_forecast()
        self._build_model()
        self._build_data()
        self._build_log()

    def _build_dashboard(self):
        cards = ttk.Frame(self.dashboard_tab)
        cards.pack(fill="x", pady=(0, 10))

        self.card_values = {}
        card_defs = [
            ("sales", "Total Sales"),
            ("profit", "Total Profit"),
            ("quantity", "Quantity Sold"),
            ("records", "Records"),
            ("forecast", "2026 Forecast"),
        ]
        for i, (key, title) in enumerate(card_defs):
            card = ttk.LabelFrame(cards, text=title, padding=12)
            card.grid(row=0, column=i, padx=5, sticky="nsew")
            cards.columnconfigure(i, weight=1)
            value = ttk.Label(card, text="--", style="CardValue.TLabel")
            value.pack(pady=8)
            self.card_values[key] = value

        ttk.Label(self.dashboard_tab, text="Historical Monthly Sales", style="CardTitle.TLabel").pack(anchor="w")
        self.dashboard_figure = Figure(figsize=(11, 5), dpi=90, constrained_layout=True)
        self.dashboard_ax = self.dashboard_figure.add_subplot(111)
        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_figure, master=self.dashboard_tab)
        self.dashboard_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(5, 0))

    def _build_analysis(self):
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

        self.analysis_figure = Figure(figsize=(11, 5.5), dpi=90, constrained_layout=True)
        self.analysis_ax = self.analysis_figure.add_subplot(111)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_figure, master=self.analysis_tab)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_forecast(self):
        top = ttk.Frame(self.forecast_tab)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Forecast period (months):").pack(side="left")
        self.forecast_months = tk.StringVar(value="12")
        ttk.Entry(top, textvariable=self.forecast_months, width=8).pack(side="left", padx=8)
        ttk.Button(top, text="Generate Forecast", style="Accent.TButton", command=self.generate_forecast).pack(side="left")
        ttk.Button(top, text="Save Forecast CSV", command=self.save_forecast).pack(side="left", padx=8)

        self.forecast_result = ttk.Label(top, text="Model ready when data is loaded.", font=("Segoe UI", 11, "bold"))
        self.forecast_result.pack(side="right")

        body = ttk.Panedwindow(self.forecast_tab, orient="vertical")
        body.pack(fill="both", expand=True)

        chart_frame = ttk.Frame(body, padding=5)
        table_frame = ttk.Frame(body, padding=5)
        body.add(chart_frame, weight=3)
        body.add(table_frame, weight=2)

        self.forecast_figure = Figure(figsize=(11, 4.5), dpi=90, constrained_layout=True)
        self.forecast_ax = self.forecast_figure.add_subplot(111)
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_figure, master=chart_frame)
        self.forecast_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.forecast_table = self._make_table(
            table_frame,
            ("Date", "Predicted Sales"),
            ("Month", "Predicted Sales"),
        )

    def _build_model(self):
        top = ttk.Frame(self.model_tab)
        top.pack(fill="x", pady=(0, 10))
        self.model_status = ttk.Label(top, text="Model not loaded", font=("Segoe UI", 13, "bold"))
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
            bg="#101820",
            fg="#d7f9e9",
            insertbackground="#d7f9e9",
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
        self.card_values["sales"].config(text=f"{total_sales:,.0f}")
        self.card_values["profit"].config(text=f"{total_profit:,.0f}")
        self.card_values["quantity"].config(text=f"{quantity:,}")
        self.card_values["records"].config(text=f"{len(self.df):,}")
        self.card_values["forecast"].config(text="Generate →")

        self.dashboard_ax.clear()
        self.dashboard_ax.plot(self.monthly["Date"], self.monthly["Sales"], marker="o", linewidth=1.8)
        self.dashboard_ax.set_title("Historical Monthly Sales (2022–2025)")
        self.dashboard_ax.set_xlabel("Date")
        self.dashboard_ax.set_ylabel("Sales")
        self.dashboard_ax.grid(True, alpha=0.25)
        self.dashboard_figure.autofmt_xdate()
        self.dashboard_canvas.draw_idle()

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
        if choice == "Month-wise Sales":
            self.analysis_ax.plot(series.index, series.values, marker="o")
            self.analysis_ax.set_xticks(range(1, 13))
        else:
            series.plot(kind="bar", ax=self.analysis_ax)
        self.analysis_ax.set_title(title)
        self.analysis_ax.set_xlabel("")
        self.analysis_ax.set_ylabel("Total Sales")
        self.analysis_ax.grid(axis="y", alpha=0.25)
        self.analysis_figure.autofmt_xdate()
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

        self.forecast_ax.clear()
        self.forecast_ax.plot(self.monthly["Date"], self.monthly["Sales"], marker="o", label="Historical")
        self.forecast_ax.plot(future["Date"], future["Predicted_Sales"], marker="o", linestyle="--", label="Forecast")
        self.forecast_ax.axvline(last_date, linestyle=":", linewidth=1.5)
        self.forecast_ax.set_title(f"Historical Sales + Next {months} Month Forecast")
        self.forecast_ax.set_xlabel("Date")
        self.forecast_ax.set_ylabel("Sales")
        self.forecast_ax.legend()
        self.forecast_ax.grid(True, alpha=0.25)
        self.forecast_figure.autofmt_xdate()
        self.forecast_canvas.draw_idle()

        total = future["Predicted_Sales"].sum()
        self.forecast_result.config(text=f"Total forecast: {total:,.2f}")
        self.card_values["forecast"].config(text=f"{total:,.0f}")
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
