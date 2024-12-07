import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch(city_id: int, timestamp: datetime, offset=0):
    start = int((timestamp - timedelta(hours=offset)).timestamp())
    url = (f"https://history.openweathermap.org/data/2.5/history/city?type=hour"
           f"&id={city_id}&start={start}&cnt={24}&appid={os.environ['OPEN_WEATHER_API_KEY']}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data for {timestamp.strftime('%y-%m-%d')}: {response.json()}")
        return None

    return {start: response.json()}


def _fetch_multiple(city_id: int, timestamps: list[datetime], offset=0):
    if os.path.exists("data/data.json"):
        with open("data/data.json") as file:
            data = json.load(file)
            return data

    data = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_fetch, city_id, timestamp, offset): timestamp
            for timestamp in timestamps
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                data.append(result)
                print(f"Done: {futures[future].strftime('%y-%m-%d')}")

    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/data.json", "w") as file:
        json.dump(data, file)


def get_feature_names():
    return np.array(["timestamp", "temp", "temp_min", "temp_max",
                     "feels_like", "pressure", "humidity", "windx",
                     "windy", "rain", "snow", "clouds"])


def _extract_data_point(dp: dict):
    if dp["cnt"] != 24:
        raise ValueError

    result = np.empty(shape=(24, 12))
    dp["list"].sort(key=lambda v: v["dt"])

    for i, item in enumerate(dp["list"]):
        main, wind = item.get("main", {}), item.get("wind", {})
        rain, snow = item.get("rain", {}), item.get("snow", {})
        clouds, weather = item.get("clouds", {}), item.get("weather", {})
        deg = wind.get("deg", np.random.randint(0, 36) * 10) * np.pi / 180

        result[i] = [
            item.get("dt"), main.get("temp"), main.get("temp_min"),
            main.get("temp_max"), main.get("feels_like"), main.get("pressure"),
            main.get("humidity"), wind.get("speed", 0) * np.cos(deg),
                                  wind.get("speed", 0) * np.sin(deg), rain.get("1h", 0),
            snow.get("1h", 0), clouds.get("all", 0),
        ]

    return result


def get_dataset(*args) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists("data/data.json"):
        _fetch_multiple(*args)

    with open("data/data.json") as file:
        days = json.load(file)["data"]
        days.sort(key=lambda v: tuple(v.keys())[0])

    data = []
    start, end = 0, 4
    while end < len(days):
        results = np.empty(shape=(24 * 3, 12))
        window = [tuple(item.values())[0] for item in days[start: end]]

        try:
            for i, item in enumerate(window[0: -1]):
                results[24 * i: 24 * (i + 1), :] = _extract_data_point(item)
            label = _extract_data_point(window[-1]).mean(axis=0)[1]
            data.append((results, label))
        except ValueError:
            pass

        end += 1
        start += 1

    np.save("data/inputs.npy", np.array([t[0] for t in data], dtype=np.float32))
    np.save("data/labels.npy", np.array([t[1] for t in data], dtype=np.float32))

    return np.load("data/inputs.npy"), np.load("data/labels.npy")


def get_data_train_test_split(*args, shuffle=True):
    nov_1 = 327
    inputs, labels = get_dataset(*args)

    x_test, y_test = inputs[nov_1:], labels[nov_1:]
    x_train, y_train = inputs[:nov_1], labels[:nov_1]

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0) + 1e-7

    x_test = (x_test - train_mean) / train_std
    x_train = (x_train - train_mean) / train_std

    test_indices = np.arange(len(x_test))
    train_indices = np.arange(len(x_train))

    if shuffle:
        np.random.shuffle(test_indices)
        np.random.shuffle(train_indices)

    return (x_train[train_indices], y_train[train_indices],
            x_test[test_indices], y_test[test_indices])
