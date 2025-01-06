from pathlib import Path

import pandas as pd


def main():
    csv_path = Path(__file__).parent / "tmp" / "data" / "epassi_statistics.csv"
    data = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True,
    )
    data = data.loc[data["Benefit type"] == "(Lounas)"].copy()
    data = data.loc[data.index.weekday < 5]
    place_names = data["Toimipiste"].unique()
    place_names_new_map = {
        place_name: f"Lunch place {letter}"
        for place_name, letter in zip(place_names, "ABCDEFGHIJKL")
    }
    data["Toimipiste"] = data["Toimipiste"].apply(
        lambda s: place_names_new_map.get(s, None)
    )
    data.dropna(subset=["Toimipiste"], inplace=True)
    data.to_csv(csv_path.parent / "anonymous_version.csv", quoting=1)


if __name__ == "__main__":
    main()
