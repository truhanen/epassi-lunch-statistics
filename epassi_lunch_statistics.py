import argparse
import calendar
import re
from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt


def get_color_cycle() -> list[tuple[float]]:
    return list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)[1::2]


def get_place_counts(data: pd.DataFrame) -> pd.DataFrame:
    place_counts = data["Toimipiste"].value_counts().to_frame()
    place_counts["percentage"] = place_counts / place_counts.sum() * 100.0
    # Sort places with equal counts alphabetically
    place_counts["place"] = place_counts.index
    place_counts.sort_values(
        by=["count", "place"], ascending=[False, True], inplace=True
    )
    place_counts.drop("place", axis=1)
    return place_counts


def get_place_color_map(data: pd.DataFrame) -> dict[str, tuple[float]]:
    place_counts = get_place_counts(data=data)
    high_count_places = list(place_counts.index.values[:10])
    colors = get_color_cycle()[: len(high_count_places) + 1]
    place_color_map = dict()
    for place, color in zip(high_count_places, colors):
        place_color_map[place] = color
    place_color_map["Other"] = colors[len(high_count_places)]
    return place_color_map


def read_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True,
    )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data_new: pd.DataFrame = data.loc[data["Benefit type"] == "(Lounas)"].copy()
    data_new["Toimipiste"] = data_new["Toimipiste"].apply(
        lambda s: re.sub(r"[0-9]", "", s).strip()
    )
    data_new = data_new.loc[~data_new.index.duplicated(keep="first")]
    data_new = data_new.loc[data_new.index.weekday < 5]
    return data_new


def augment_data(data: pd.DataFrame):
    data["month"] = data.index.month
    data["weekday"] = data.index.weekday


def plot_place_distribution(data: pd.DataFrame, ax: plt.Axes):
    place_counts = get_place_counts(data=data)
    y_values = list(range(len(place_counts["percentage"])))
    ax.barh(y=y_values, height=0.8, width=place_counts["count"])
    ax.set_yticks(y_values, place_counts.index)
    ax.set_xlabel("Overall count")
    ax.grid(alpha=0.3)
    ax_percentage = ax.twiny()
    ax_xmax_as_percentage = ax.get_xlim()[1] / place_counts["count"].sum() * 100.0
    ax_percentage.set_xlim(0, ax_xmax_as_percentage)
    ax_percentage.set_xlabel("Overall percentage")
    ax.spines["right"].set_visible(False)


def plot_place_distributions_by_month(data: pd.DataFrame, ax: plt.Axes):
    place_counts = get_place_counts(data=data)
    high_count_places = list(place_counts.index.values[:10])

    place_percentages_by_month: pd.Series = (
        data[["month", "Toimipiste"]].groupby("month").value_counts(normalize=True)
    ) * 100.0
    place_percentages_by_month.sort_index(level=["Toimipiste", "month"], inplace=True)
    place_percentages_by_month.index = place_percentages_by_month.index.reorder_levels(
        ["Toimipiste", "month"]
    )

    place_percentages_by_month = place_percentages_by_month.loc[
        place_percentages_by_month.index.get_level_values("Toimipiste").isin(
            high_count_places
        )
    ]
    month_min = min(data.index.month)
    month_max = max(data.index.month)
    month_range = list(range(month_min, month_max + 1))
    index_filled = pd.MultiIndex.from_product(
        [
            place_percentages_by_month.index.get_level_values(
                "Toimipiste"
            ).drop_duplicates(),
            month_range,
        ],
        names=["Toimipiste", "month"],
    )
    place_percentages_by_month_filled = pd.Series(
        data=[0.0] * len(index_filled), index=index_filled, name="proportion"
    )
    place_percentages_by_month_filled.loc[place_percentages_by_month.index] = (
        place_percentages_by_month
    )

    place_color_map = get_place_color_map(data=data)
    for i, month in enumerate(month_range):
        percentages = []
        colors = []
        for place in high_count_places:
            percentage = place_percentages_by_month_filled.loc[(place, month)]
            if percentage > 0.0:
                percentages.append(percentage)
                colors.append(place_color_map[place])
        if sum(percentages) < 100.0:
            percentages.append(100.0 - sum(percentages))
            colors.append(place_color_map["Other"])
        ax_month_x = i % 4 * 0.25
        ax_month_y = 1 - (i // 4 + 1) * 0.3
        ax_month: plt.Axes = ax.inset_axes(
            bounds=(ax_month_x, ax_month_y, 1 / 4, 1 / 4)
        )
        ax_month.pie(x=percentages, colors=colors)
        ax_month.set_xlabel(calendar.month_name[month])
    ax.axis("off")


def plot_place_distributions_by_weekday(data: pd.DataFrame, ax: plt.Axes):
    place_counts = get_place_counts(data=data)
    high_count_places = list(place_counts.index.values[:10])

    place_percentages_by_weekday: pd.Series = (
        data[["weekday", "Toimipiste"]].groupby("weekday").value_counts(normalize=True)
    ) * 100.0
    place_percentages_by_weekday.sort_index(
        level=["Toimipiste", "weekday"], inplace=True
    )
    place_percentages_by_weekday.index = (
        place_percentages_by_weekday.index.reorder_levels(["Toimipiste", "weekday"])
    )

    place_percentages_by_weekday = place_percentages_by_weekday.loc[
        place_percentages_by_weekday.index.get_level_values("Toimipiste").isin(
            high_count_places
        )
    ]
    weekday_range = list(range(0, 5))
    index_filled = pd.MultiIndex.from_product(
        [
            place_percentages_by_weekday.index.get_level_values(
                "Toimipiste"
            ).drop_duplicates(),
            weekday_range,
        ],
        names=["Toimipiste", "weekday"],
    )
    place_percentages_by_weekday_filled = pd.Series(
        data=[0.0] * len(index_filled), index=index_filled, name="proportion"
    )
    place_percentages_by_weekday_filled.loc[place_percentages_by_weekday.index] = (
        place_percentages_by_weekday
    )

    place_color_map = get_place_color_map(data=data)
    add_label_for_other = False
    for i, weekday in enumerate(weekday_range):
        percentages = []
        colors = []
        for place in high_count_places:
            percentage = place_percentages_by_weekday_filled.loc[(place, weekday)]
            if percentage > 0.0:
                percentages.append(percentage)
                colors.append(place_color_map[place])
        if sum(percentages) < 100.0:
            percentages.append(100.0 - sum(percentages))
            colors.append(place_color_map["Other"])
            add_label_for_other = True
        ax_weekday_x = i % 3 * (1 / 3)
        ax_weekday_y = 0.795 - (i // 3 + 1) * 0.35
        ax_weekday: plt.Axes = ax.inset_axes(
            bounds=(ax_weekday_x, ax_weekday_y, 1 / 3, 1 / 3)
        )
        ax_weekday.pie(x=percentages, colors=colors)
        ax_weekday.set_xlabel(calendar.day_name[weekday])
    places_with_label = high_count_places.copy()
    if add_label_for_other:
        places_with_label.append("Other")
    for place in places_with_label:
        ax.bar(0, 0, 0, color=place_color_map[place], label=place)
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), ncol=2, fontsize="small")
    ax.axis("off")


def plot_place_occurrences(data: pd.DataFrame, ax: plt.Axes):
    place_counts = get_place_counts(data=data)
    y_values = list(range(len(place_counts)))
    for place, y_value in zip(place_counts.index, y_values):
        data_of_place = data.loc[data["Toimipiste"] == place].copy()
        data_of_place.sort_index(inplace=True)
        time_ranges = [
            (pd.Timestamp(index_value), timedelta(days=1))
            for index_value in data_of_place.index.values
        ]
        y_range = (y_value - 0.4, 0.8)
        ax.broken_barh(xranges=time_ranges, yrange=y_range)
    ax.set_yticks(y_values, place_counts.index)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Occurrences")
    ax.tick_params(left=False)


def plot_data(data: pd.DataFrame, figure_path: Path):
    figure_path.parent.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(
        2, 2, figsize=(14, 14), constrained_layout=True, sharey="row"
    )
    (
        ax_place_distribution,
        ax_place_occurrences,
        ax_place_distributions_by_weekday,
        ax_place_distributions_by_month,
    ) = axs.flat

    plot_place_distribution(data=data, ax=ax_place_distribution)
    plot_place_occurrences(data=data, ax=ax_place_occurrences)
    plot_place_distributions_by_weekday(data=data, ax=ax_place_distributions_by_weekday)
    plot_place_distributions_by_month(data=data, ax=ax_place_distributions_by_month)

    fig.savefig(figure_path, dpi=200)


def analyze(csv_path: Path, figure_path: Path):
    data = read_data(csv_path=csv_path)
    data = preprocess_data(data=data)
    augment_data(data=data)
    plot_data(data=data, figure_path=figure_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path of the input CSV file")
    parser.add_argument("output_path", type=Path, help="Path for the output figure")
    arguments = parser.parse_args()

    analyze(csv_path=arguments.csv_path, figure_path=arguments.output_path)


if __name__ == "__main__":
    main()
