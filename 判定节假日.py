# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:41:02 2018

@author: User
"""

# -*- coding:utf-8 -*-    
import json  
import urllib2  
date = "20180422"  
server_url = "http://www.easybots.cn/api/holiday.php?d="  
  
vop_response = urllib.request.urlopen(server_url+date)  
  
vop_data= json.loads(vop_response.read())  
  
print vop_data  
  
if vop_data[date]=='0':  
    print "this day is weekday"  
elif vop_data[date]=='1':  
    print 'This day is weekend'  
elif vop_data[date]=='2':  
    print 'This day is holiday'  
else:  
    print 'Error'