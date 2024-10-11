#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2020
# E-mail: ivar@usp.br

import bcrypt
from bson.objectid import ObjectId

from vx.radpleura.Settings import *
#from vx.media.Access import *
from vx.com.py.database.MongoDB import *


class User:
    @staticmethod
    def newUser(userdata):
        rest = ""
        rs = MongoDB.find(DBS.DBMEDIA, "user", {"email": userdata["email"]})
        if len(rs)==0:
            np = userdata["password"].encode("utf-8");
            np = bcrypt.hashpw(np, bcrypt.gensalt()).decode('utf-8')
            userdata["password"] = np;
            rest = MongoDB.insert(DBS.DBMEDIA, "user", userdata)
            rest = str(rest)
        return rest

    @staticmethod
    def setAdmin(userdata):
        rest = 0
        rs = MongoDB.find(DBS.DBMEDIA, "user", {"email": userdata["email0"],"adminid": 1})
        if len(rs)==1:
            MongoDB.update( DBS.DBMEDIA,"user",
                            {"email": userdata["email1"]},
                            {"adminid": userdata["adminid"]})
            rest = 1;
        return rest
    
    @staticmethod
    def getUsers():
        rest = []
        userdata = MongoDB.find(DBS.DBMEDIA, "user", {})
        if len(userdata) > 0:
            for row in userdata:
                rs = {}
                rs["_id_user"] = str(row["_id"])
                rs["name"] = row["name"]
                rs["email"] = row["email"]
                rs["adminid"] = 0
                if "adminid" in row:
                    rs["adminid"] = row["adminid"]

                if rs["_id_user"] != Settings.ROOTID:
                    rest.append(rs)

        return rest

    @staticmethod
    def login(us, ps):
        response = []
        ps = ps.encode("utf-8")
        #print("ps", us, ps)
        #re = list(MongoDB.find(DBS.DBMEDIA, "user", {}))
        re = MongoDB.find(DBS.DBMEDIA, "user", {"email":us})
        print("re", re)
        
        if len(re)==1:
            for r in re:
                hd = r["password"].encode("utf-8")
                #print("ps, r['password']",ps, hd)
                if bcrypt.checkpw(ps, hd):

                    # re['adminid'] = 0

                    response = re
                    if not "adminid" in response[0]:
                        #print("responsexxx", response)
                        response[0]["adminid"] = "0";
                    
        return response

    @staticmethod
    def changePassword(app, iduser):
        iduser = iduser.decode("utf-8")
        ps = app.argms["password"].encode("utf-8");
        np = app.argms["newpassword"].encode("utf-8");

        np = bcrypt.hashpw(np, bcrypt.gensalt()).decode('utf-8')

        rt = 0
        re = MongoDB.find(DBS.DBMEDIA, "user",
                {"_id":ObjectId(iduser)})

        if len(re):
            for u in re:
                if bcrypt.checkpw(ps, u["password"].encode("utf-8")):
                    MongoDB.update(DBS.DBMEDIA, "user", 
                            {"_id":ObjectId(iduser)}, {"password":np})
                    rt = 1

        return {"response":rt};


    @staticmethod
    def rootinit():
        # root user
        userdata = {
                "name":"root",
                "fono":"root",
                "addres":"root",
                "email": "root",
                "password": "admin",
                "adminid":1,
                };
        rest = ""
        rs = list(MongoDB.find(DBS.DBMEDIA, "user", {"email": userdata["email"]}));
        if len(rs)==0:
            np = userdata["password"].encode("utf-8");
            np = bcrypt.hashpw(np, bcrypt.gensalt()).decode('utf-8')
            userdata["password"] = np;
            rest = MongoDB.insert(DBS.DBMEDIA, "user", userdata)
            Settings.ROOTID = str(rest)
        elif len(rs)==1:
            Settings.ROOTID = str(rs[0]["_id"])
        # END root user ############

        #local user
        userdata = {
                "name":"localuser",
                "fono":"localuser",
                "addres":"localuser",
                "email": "localuser",
                "password": "admin",
                "adminid":0,
                };
        rest = ""
        rs = list(MongoDB.find(DBS.DBMEDIA, "user", {"email": userdata["email"]}));
        if len(rs)==0:
            np = userdata["password"].encode("utf-8");
            np = bcrypt.hashpw(np, bcrypt.gensalt()).decode('utf-8')
            userdata["password"] = np;
            rest = MongoDB.insert(DBS.DBMEDIA, "user", userdata)
        # END local user ############

    """ 
    @staticmethod
    def rootread():
        fi = open("./vx/media/config/init.obj", 'r')
        Settings.ROOTID = fi.readline().strip()
        fi.close() """


