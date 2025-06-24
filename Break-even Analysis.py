# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 09:00:24 2025

@author: Katie
"""

import matplotlib.pyplot as plt 
import numpy as np 

units = np.linspace(1, 15_000, 1000)

# === 1. Non-recurring cost line ===
nonrecurring_total = 466_277_000
nonrecurring_line = np.full_like(units, nonrecurring_total)

# === 2. Recurring + NRC line ===
recurring_unit_cost = 64_936.89
recurring_line = recurring_unit_cost * units
total_cost_line = recurring_line + nonrecurring_line

# === 3. Price line  ===
unit_price = 99_575
price_line = unit_price * units

# === Optional: Break-even marker ===
profit = price_line - total_cost_line
try:
    breakeven_index = np.where(profit >= 0)[0][0]
    breakeven_units = units[breakeven_index]
    breakeven_profit = profit[breakeven_index]
    breakeven_marker = True
except IndexError:
    breakeven_marker = False

# === Plot ===
plt.figure(figsize=(9, 5))
plt.plot(units, nonrecurring_line, label="Non-Recurring Costs", color="orange")
plt.plot(units, total_cost_line, label="Total Cost (Recurring + NRC)", color="red")
plt.plot(units, price_line, label="Revenue (Price × Units)", color="green")

if breakeven_marker:
    plt.axvline(x=breakeven_units, color="gray", linestyle="--")
    plt.text(breakeven_units, breakeven_profit,
             f'Break-even at {int(breakeven_units)} units',
             verticalalignment='bottom', horizontalalignment='right',
                 fontsize=9, color='black',
                 bbox=dict(facecolor='white', edgecolor='gray'))

plt.xlabel("Units Produced")
plt.ylabel("Total Value ($)")
plt.title("Owl-22 Break-even Cost & Revenue Analysis")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()