# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:45:03 2017

@author: dhavalma
"""

# importing Flask
from flask import Flask, jsonify, render_template, request
import json
# import tic tac toe game 
from Main import *
import numpy as np
sess = tf.InteractiveSession()

x , prediction, _ = createNetwork()

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("model")

if checkpoint and checkpoint.model_checkpoint_path:
    s = saver.restore(sess,checkpoint.model_checkpoint_path)
    print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")
graph = tf.get_default_graph()

app = Flask(__name__, template_folder="./")

@app.route('/')
def index():
    return render_template('index.html')

def  bestmove(input):
    global graph
    with graph.as_default():
        data = (sess.run(tf.argmax(prediction.eval(session = sess,feed_dict={x:[input]}),1)))
    return data

@app.route('/api/ticky', methods=['POST'])
def ticky_api():
    data = request.get_json()
    data = np.array(data['data'])
    data = data.tolist()
    #print('data is ')
    #print(type(data))
    #print(data)
    return jsonify(np.asscalar(bestmove(data)[0]))


if __name__ == '__main__':
    app.run(host='172.25.32.55',port=8888,debug=True)
