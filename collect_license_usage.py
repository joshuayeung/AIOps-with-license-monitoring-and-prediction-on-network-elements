import time
import re
import pandas as pd
from telnetlib import Telnet
import datetime
import numpy as np

def send_command(command):
    """
    Parameters:
    command (string): the command sent to NMS

    Returns:
    result_text when success
    False when fail
    """
    tn.write(command.encode())
    time.sleep(1)
    result_text = ""
    while True:
        result = tn.read_until(b'END\r\n')
        result_text += result.decode()
        if "To be continued..." not in result.decode():
            break
    #print(result)
    if result_text.find("RETCODE = 1") == -1:
        print("run " + command + " success")
        return result_text
    else:
        print(result_text)
        return False


def login(account, password):
    """
    Return True when login success, else False
    """
    login_command = 'LGI:OP="{}", PWD="{}";\r\n'.format(account, password)
    return send_command(login_command)

def reg_ne(reg_commands):
    """
    Register a Network Element
    """
    for command in reg_commands:
        reg_success = send_command(command)
        if not reg_success:
            return reg_success
    return reg_success

def get_license(dsp_command):
    """
    Return dataframe if display license items success, else return False
    """
    result_text = send_command(dsp_command)
    if not result_text:
        return None
    results = result_text.split('\r\n')
    labels = re.split(r'\s{2,}', results[7])
    data = []

    for i in range(9,len(results)):
        if results[i].find('Number of results') != -1:
            break
        data.append(tuple(re.split(r'\s{2,}', results[i])[:-1]))

    df = pd.DataFrame.from_records(data, columns=labels)
    return df

import sqlalchemy
yourdb = sqlalchemy.create_engine('mssql+pyodbc://user:password@DATABASE_ip\\INSTANCE/datebase?driver=SQL+Server+Native+Client+11.0')

tn = Telnet(NMS_IP, NMS_Port)

if login(user, password):

    objects = A_LIST_OF_NETWORK_ELEMENTS

    for object_name in objects:
        ne = None
        reg_commands = ['REG NE:NAME={};\r\n'.format(object_name)]
        dsp_command = 'DSP LICENSERES:;\r\n'

        if reg_ne(reg_commands):
            ne = get_license(dsp_command)

            if ne is not None:
                # data cleaning
                ne[['Total Resource','Used Resource']] = ne[['Total Resource','Used Resource']].astype('int')
                ne['Usage'] = ne['Used Resource']/ne['Total Resource']
                ne.columns = ['resource_item', 'resource_name', 'total_resource', 'used_resource', 'usage']
                ne['result_time'] = datetime.datetime.now()
                ne['object_name'] = object_name
                
                ne.to_sql(name='license_table',con=yourdb, if_exists='append', index=False)
