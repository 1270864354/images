# -*- coding: utf-8 -*-
from BaseHTTPServer import BaseHTTPRequestHandler
import cgi
import numpy as np
import sys
from PIL import Image
import load_cnn
import json
reload(sys)
sys.setdefaultencoding('utf-8')



class PostHandler(BaseHTTPRequestHandler):


    def do_POST(self):
        # Parse the form data posted
        enc="UTF-8"
        form = cgi.FieldStorage(
        fp=self.rfile,
        headers=self.headers,
        environ={'REQUEST_METHOD':'POST',
          'CONTENT_TYPE':self.headers['Content-Type'],
        })
        # form = cgi.FieldStorage()


        # Begin the response
        self.send_response(200)
        self.end_headers()

        # json_str = json.loads(form.value)         
  
		# json_dict = JSONDecoder().decode(json_str)
		# for (key, value) in json_dict.items():  
		# 	data[key] = value
		# 	print key,value
        # Echo back information about what was posted in the form
        for field in form.keys():
            field_item = form[field]
            if field_item.filename:
                # The field contains an uploaded file
                print field_item.filename +'111'
                file_data = field_item.file.read()
                with open('/python/file/' + field_item.filename,'wb') as f:
                	f.write(file_data)
                label = str(load_cnn.image_one_label('/python/file/' + field_item.filename))
                dict_return ={}
                dict_return['name'] = label
                json_str = json.dumps(dict_return)
                file_len = len(file_data)
                del file_data
                self.wfile.write(json_str)


        return 

if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer
    from collections import defaultdict
    from PIL import Image
    server = HTTPServer(('0.0.0.0',80), PostHandler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()
