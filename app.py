from flask import Flask, jsonify, request
import algorithm
import numpy as np

app = Flask(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# API - Routing
# ----------------------------------------------------------------------------------------------------------------------

@app.route("/getData", methods=['POST'])
def get_data():
    plot_data, centers = algorithm.cluster_results(request.json)
    res = {}
    res['center_points'] = []
    res['data_points'] = []
    count = 1
    for cluster in plot_data.get_values():
        temp = {}
        temp['item_no'] = count
        temp['cluster'] = np.int32(cluster[0]).item()
        temp['x'] = cluster[1]
        temp['y'] = cluster[2]
        res['data_points'].append(temp)
        count = count + 1

    for j, center in enumerate(centers):
        temp = {}
        temp['x'] = center[0]
        temp['y'] = center[1]
        res['center_points'].append(temp)

    return jsonify(res)


@app.route("/getCluster", methods=['POST'])
def get_cluster():
    prediction = algorithm.find_cluster(request.json)
    temp = {}
    temp['cluster'] = np.int32(prediction).item()
    return jsonify(temp)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(7891)
    )
