# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:57:41 2017

@author: Administrator
"""

from flask import  Flask
from flask import  render_template
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
app = Flask(__name__,static_url_path='',static_folder='')
app.debug=True
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index.html')
def index2():
    return render_template('index.html')    
@app.route('/link/<link_source>')
def predict_html(link_source):
    return render_template('/link/'+link_source)
#@app.route('/link/html_prefix_name_0.html')    
#def word_html():
#    return render_template('/link/html_prefix_name_predict_0.html')    
#@app.route('/link/html_prefix_name_0.html')      
#def next_html():
#    return render_template('/link/html_prefix_name_predict_0.html')      
if __name__ == '__main__':
#    app.run('10.104.17.164',port=5000)
#    app.run()
    http_server = HTTPServer(WSGIContainer(app))
#    http_server.bind()
    http_server.listen(10000,address='10.104.17.164')
    IOLoop.instance().start()