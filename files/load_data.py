import sqlite3


def load_data():
    data = []
    with sqlite3.connect("/home/teuffenm/Desktop/waage/weight_20210217") as db:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM scaleMeasurements")

        db_data = cursor.fetchall()
        for row in db_data:
            data.append(row)

    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    data = np.asarray(data)

    # Data 2 - "deleted"
    data_deleted = np.int32(data[:, 2])
    deleted_mask = (np.asarray(data_deleted, dtype=bool))
    plt.figure(), plt.plot(data_deleted), plt.show()

    # Data 3 - date
    timestamps = np.int64(data[:, 3][deleted_mask])
    plt.figure(), plt.plot(timestamps), plt.show()
    measurement_time = []
    for timestamp in timestamps:
        measurement_time.append(datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    measurement_time = np.asarray(measurement_time)
    # Data 4 - weight reading
    weight_measurements = np.float64(data[:, 4][deleted_mask])
    plt.figure(), plt.plot(weight_measurements), plt.show()

    return timestamps / 1000 / 60, weight_measurements, measurement_time
