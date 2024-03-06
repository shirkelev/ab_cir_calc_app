import tkinter as tk
from tkinter import ttk
import statsmodels.api as sm
import numpy as np
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tools import add_constant
from statsmodels.formula.api import ols






# Replace 'your_formula' with the actual formula used for quantile regression
# Example formula: quant_formula = "ab_cir ~ week"

# Assuming you have a dictionary 'quantile_models'
# with keys as quantiles (e.g., 0.01, 0.05, ..., 1) and values as QuantRegResults objects

# Function to predict quantiles for a given pregweek and ab_cir

class PregwwekError(Exception):
    pass

def predict_quantiles(week, ab_cir, quantile_models):
    if week < 15 or week > 40:
        raise PregwwekError("Pregweek must be between 15 and 40")
    predictions = {}
    for quantile, model in quantile_models.items():
        predicted_ab_cir = model.predict(exog=dict(pregweek=week))[0]
        diff = abs(predicted_ab_cir - ab_cir)
        predictions[quantile] = diff
    closest_quantile = min(predictions, key=predictions.get)
    if closest_quantile == 0.01:
        return 0
    elif closest_quantile == 0.99 and diff >= 0.01:
        return 1
    return closest_quantile



# GUI setup
class QuantileCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.data = None
        self.models = None
        self.root.title("Quantile Calculator")
        self.label_week = None
        self.entry_week = None
        self.label_ab_cir = None
        self.entry_ab_cir = None
        self.calculate_button = None
        self.result_label = None
        
    def create_widgets(self):
        # Entry widgets for user input
        # Entry widgets for user input
        self.label_week = ttk.Label(root, text="Pregweek:")
        self.entry_week = ttk.Entry(root, state= "disabled")
        self.label_ab_cir = ttk.Label(root, text="AB Cir:")
        self.entry_ab_cir = ttk.Entry(root, state= "disabled")
        self.canvas = None

        # Button to calculate closest quantile
        self.calculate_button = ttk.Button(root, text="Calculate", command=self.calculate_quantile, state= "disabled")

        # Label to display the result
        self.result_label = ttk.Label(root, text="Loading...")
        # Grid layout
        self.label_week.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_week.grid(row=0, column=1, padx=5, pady=5)
        self.label_ab_cir.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_ab_cir.grid(row=1, column=1, padx=5, pady=5)
        self.calculate_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.result_label.grid(row=3, column=0, columnspan=2)
        
        
    def calculate_quantile(self):
        try:
            # Remove plot if exists
            for widget in self.root.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    widget.get_tk_widget().destroy()
                    
            week = float(self.entry_week.get())
            ab_cir = float(self.entry_ab_cir.get())
            closest_quantile = predict_quantiles(week, ab_cir, self.models)
            closest_quantile = round(closest_quantile, 2)
            self.result_label.config(text=f"""The closest quantile to your parameter is {closest_quantile*100}% \n Which means that {closest_quantile * 100}% of the fetuses on pregnant week {week} have less than {ab_cir} abdominal circumference.""")
            # Set the result label display to the center
            self.result_label.grid(row=3, column=0, columnspan=2)
            self.plot_graph({'pregweek': week, 'ab_cir': ab_cir})
        except ValueError:
            self.result_label.config(text="Invalid input. Please enter numeric values.")
        except PregwwekError as e:
            self.result_label.config(text=str(e))
    ## Function to plot the graph of quantile [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] as function of pregweek and the given dot on the app
    def plot_graph(self, given_point):
        
        # Create a figure with the maximal size possible
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Quantiles to plot
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        # Create a color palette
        colors = sns.color_palette("Oranges", len(quantiles))
        # Plot quantiles as function of pregweek
        pregweek_values = np.arange(15, 41)
        for quantile in quantiles:
            quantile_values = []
            for week in pregweek_values:
                predicted_ab_cir = self.models[quantile].predict(exog=dict(pregweek=week))[0]
                quantile_values.append(predicted_ab_cir)
            ax.plot(pregweek_values, quantile_values, label=f"Q {quantile}", color=colors[quantiles.index(quantile)])
        
        # Plot the given point
        ax.plot(given_point['pregweek'], given_point['ab_cir'], 'ro', label="Given Point")

        # Set labels and title
        ax.set_xlabel("Pregweek")
        ax.set_ylabel("AB Cir")
        ax.set_title("Quantile Regression")

        # Add legend
        ax.legend()

        # Create a canvas and toolbar
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)
    
    def set_models(self, models):
        self.models = models
         # Entry widgets for user input
        self.entry_week.state(["!disabled"])
        self.entry_ab_cir.state(["!disabled"])
        self.calculate_button.state(["!disabled"])
        self.result_label.config(text = "")
        # Label to display the result
        self.result_label = ttk.Label(root, text="")

    def gather_data(self):
        df=pd.read_csv("rawdata.csv")
        df = df.dropna(subset=['pregweek', 'ab_cir'])
        df.loc[df['ab_cir'] > 500, 'ab_cir'] /= 10
        df.loc[df['ab_cir'] < 40, 'ab_cir'] *= 10
        df = df[(df['pregweek'] >= 15) & (df['pregweek'] <= 40)]
        df = df[(df['preg_num'] <= 15)]
        self.data = df
    
    def train_models(self):
        quantile_models = {}
        model_formula =  'ab_cir ~ pregweek + I(pregweek**2) + np.log(pregweek) + np.sqrt(pregweek)'
        quantiles_to_calc = [0.01, 0.03, 0.05, 0.07, 0.9, 0.92, 0.95, 0.97, 0.99]
        quantiles_to_calc += [tau / 100 for tau in range(10, 90, 5)]
        # print(quantiles_to_calc)
        self.result_label.config(text = "Gathring data and training models is done")
        for tau in quantiles_to_calc:
            quant_mod = sm.QuantReg.from_formula(model_formula, data=self.data)
            quant_fit = quant_mod.fit(q=tau)
            quantile_models[tau] = quant_fit
        print("Done")
        self.set_models(quantile_models)

    def run_main_loop(self):
        self.create_widgets()
        self.gather_data()
        self.train_models()
        
        
# Initialize and run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = QuantileCalculatorApp(root)
    root.after(0, app.run_main_loop)
    root.mainloop()
    print("Done")
    
    
