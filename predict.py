import numpy as np
import seaborn as sns
from calendar import Calendar
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

from keras import models, mixed_precision
mixed_precision.set_global_policy("mixed_float16")
from dataset import get_data_train_test_split, get_feature_names

model_id = 16
model = models.load_model(f"models/model_{model_id}.keras", compile=False)

_, _, x_test, y_test = get_data_train_test_split()

mask = [1, 5, 6, 7, 8, 9, 10, 11]
feature_names = get_feature_names()[mask]
x_test = x_test[:, :, mask]

# %% Predictions on 2D plot
k_to_c = -273.15  # Convert between kelvin (K°) and celsius (C°)
green = np.array([[74, 130, 99]]) / 255
lilac = np.array([[168, 85, 182]]) / 255

# Predict and evaluate
y_pred = model.predict(x_test, batch_size=1).flatten()
plt.scatter(y_test + k_to_c, y_pred + k_to_c, c=lilac, alpha=0.6, label="Data Points")

loss, mse, mae = model.evaluate(x_test, y_test, batch_size=1)
print(f"Test MSE: {mse}, MAE: {mae}")

# Fit a regression line
regressor = LinearRegression()
regressor.fit(y_test.reshape(-1, 1), y_pred)
y_fit = regressor.predict(np.array([min(y_test), max(y_test)]).reshape(-1, 1))

m, b = regressor.coef_[0], regressor.intercept_
x_range = np.linspace(min(y_test) - 1.5, max(y_test) + 1.5, 100).reshape(-1, 1) + k_to_c
plt.plot(x_range, m * (x_range - k_to_c) + b + k_to_c,
         color=lilac + 0.20, linewidth=2, label="Prediction")

# Add identity line
identity_line = np.linspace(min(y_test) - 1.5, max(y_test) + 1.5, 200) + k_to_c
plt.plot(identity_line, identity_line, color=green, linewidth=2, linestyle='--', label="Identity")

# Finalize plot
plt.xlabel("Actual Temperatures (C°)")
plt.ylabel("Predicted Temperatures (C°)")
plt.title("Actual vs. Predicted Temperatures")

plt.legend()
plt.tight_layout()
plt.savefig(f"figures/predictions_{model_id}.png", dpi=600)


# %% Permutation importance plot

class ModelPermute:
    def __init__(self, model_p: models.Model, original_shape):
        self.model = model_p
        self.original_shape = original_shape

    def fit(self, x, y):
        x_reshaped = x.reshape(-1, *self.original_shape[1:])
        self.model.fit(x_reshaped, y, batch_size=1)
        return self

    def predict(self, x):
        x_reshaped = x.reshape(-1, *self.original_shape[1:])
        return self.model.predict(x_reshaped, batch_size=1)


# Perform permutation importance
reshaped_model = ModelPermute(model, x_test.shape)
r = permutation_importance(reshaped_model, x_test.reshape(-1, x_test.shape[2]),
                           y_test, n_repeats=30, random_state=95, scoring="r2")


def clip_outliers(data, low=15, high=85):
    lower_bound = np.percentile(data, low, axis=1, keepdims=True)
    upper_bound = np.percentile(data, high, axis=1, keepdims=True)
    return np.clip(data, lower_bound, upper_bound)


# Sort features by the mean absolute importance after clipping
importances_clipped = clip_outliers(r.importances, low=15, high=85)
importances_mean = np.mean(np.abs(importances_clipped), axis=1)
sorted_indices = np.argsort(importances_mean)
importances_sorted = importances_clipped[sorted_indices]
feature_names_sorted = [feature_names[i] for i in sorted_indices]

# Plotting the boxplot with clipped importance values
plt.figure(figsize=(10, 6))
plt.boxplot(importances_sorted.T, vert=False,
            tick_labels=feature_names_sorted,
            patch_artist=True, showmeans=True,
            boxprops=dict(color=green, facecolor=green),
            capprops=dict(color=green, linewidth=1.5),
            whiskerprops=dict(color=green, linewidth=1.5),
            medianprops=dict(color=lilac + 0.05, linewidth=3),
            meanprops=dict(marker="d", markerfacecolor=lilac + 0.08,
                           markeredgecolor="white", markeredgewidth=0.3)
            )

plt.xlabel('Decrease in Accuracy Score')
plt.title('Permutation Importances (Test Set)')
plt.axvline(x=0, color='black', linestyle='--', lw=1)

plt.tight_layout()
plt.savefig(f"figures/permutations_{model_id}.png", dpi=600)

# %% Error calendar view plot
custom_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

# Example prediction errors for 30 days (replace with actual data)
daily_errors = [y_pred[i] - y_test[i] for i in range(len(y_test))]
daily_errors = np.array(daily_errors)

norm = plt.Normalize(-round(max(daily_errors)), round(max(daily_errors)))

# Generate a calendar for November 2024
cal = Calendar()
year, month = 2024, 11
month_days = cal.monthdayscalendar(year, month)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f"Daily Prediction Errors - November {year}",
             fontsize=18, fontweight='bold', pad=20)

# Plot each day in the calendar
day_counter = 0
for week_idx, week in enumerate(month_days):
    week_idx += 1
    for day_idx, day in enumerate(week):
        if day == 0:
            continue

        error = daily_errors[day_counter]
        color = custom_cmap(norm(error))

        # Add a colored rectangle for each day
        ax.add_patch(plt.Rectangle((day_idx, -week_idx), 1, 1,
                                   facecolor=color, edgecolor='black', lw=0.5))

        # Add the date and error value
        ax.text(day_idx + 0.5, -week_idx + 0.65, f"{day}",
                ha="center", va="center", fontsize=12, fontweight='bold')
        ax.text(day_idx + 0.5, -week_idx + 0.3, f"{error:.2f}",
                ha="center", va="center", fontsize=10, color="black")

        day_counter += 1

# Customize the axes
ax.set_xticks(np.arange(0.5, 7.5, 1))
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                   fontsize=10, fontweight='bold')
ax.set_yticks(range(-len(month_days), 0))
ax.set_yticklabels([])

ax.set_xlim(0, 7)
ax.set_ylim(-len(month_days), 0)
ax.set_aspect("equal")

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1, aspect=40)
cbar.set_label("Prediction Error (C°)", fontsize=12)

# # Add gridlines for visual clarity
for x_i in range(7):
    ax.axvline(x_i, color='black', linewidth=0.8, alpha=0.5)
for y_i in range(-len(month_days), 1):
    ax.axhline(y_i, color='black', linewidth=0.8, alpha=0.5)

plt.savefig(f"figures/calendar_{model_id}.png", dpi=600)
