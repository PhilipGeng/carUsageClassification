from dateutil import parser
import numpy as np
import itertools
import operator
from scipy import stats
import matplotlib.pyplot as plt
import collections

def reject_outliers(data,threshold, m=2):
    data = np.array(data)
    return data[abs(data - stats.mode(data)[0]) < m *max(np.std(data),threshold)]

def reject_outliers_mean(data,m=2):
    data = np.array(data)
    return data[abs(data - np.mean(data)) < m *np.nanstd(data)]


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

#vinList = ["S029328","S056452","S018514","S071917","S087986","S030252","S061996","S041762","S063813","S034824", "S031375","S064587","S014605","S015525","S026409","S060512","S061316","S072170","S092211","S070756", "S020307","S048754","S014834","S065263","S064495","S050233","S058412","S060583","S015156","S059648", "S048735","S059642","S066648","S060949","S061178","S033569","S015405","S025873","S034806","S062379", "S018739","S091058","S015141","S043764","S061205","S064476","S087971","S026864","S048330","S065097"]
#vinList = ["S070444", "S026409", "S061996", "S062379", "S057382", "S059642", "S031642", "S037398", "S029328", "S014566", "S066648", "S062132", "S065263", "S032895", "S015156", "S092330", "S084227", "S028163", "S031375", "S071917", "S014834", "S034806", "S014605", "S018514", "S014808", "S061205", "S030207", "S064255", "S062250", "S027644", "S026864", "S091198", "S092211", "S030252", "S082656", "S064326", "S062003", "S063252", "S020307", "S064476", "S019542", "S015525", "S070823", "S048330", "S041762", "S018804", "S031378", "S060493", "S015141", "S059648", "S062610", "S066659", "S063813", "S074547", "S048735", "S015405", "S064495", "S072170", "S034824", "S076862", "S061178", "S061228", "S045607", "S058412", "S017266", "S016169", "S061316", "S031519", "S056452", "S064587", "S060583", "S087971", "S070756", "S023899", "S033569", "S057207", "S035077", "S091058", "S019509", "S018739", "S060514", "S060512", "S043533", "S034059", "S070599", "S048754", "S043764", "S050234", "S075660", "S063770", "S061627", "S050233", "S063352", "S025873", "S087986", "S065097", "S046615", "S061404", "S060949", "S047271"]
#vinList = ["S070444"] #File Query here
vinList = ["S069183", "S074220", "S076180", "S084215" ,"S084389", "S085985", "S088390", "S090534", "S093039", "S093049", "S093681", "S094566", "S069898", "S070599", "S074128", "S074524", "S075053", "S085590", "S086017", "S088316", "S088360", "S089216", "S090606", "S091190", "S092371", "S092635", "S093025", "S069165", "S069873", "S071328", "S071829", "S074095", "S076179", "S076211", "S077224", "S077910", "S084142", "S086112", "S087987", "S091203", "S092030", "S092526", "S092701", "S093041", "S094024", "S070317", "S070481", "S070650", "S070823", "S072629", "S074110", "S074221", "S074578", "S075406", "S075729", "S084649", "S084669", "S086000", "S086346", "S086909", "S088343", "S088580", "S089217", "S089822", "S091266", "S092308", "S093317", "S093371", "S093665", "S060951", "S061391", "S062032", "S064235", "S064245", "S064597", "S065866", "S060868", "S061174", "S061187", "S061199", "S062963", "S063812", "S064478", "S064493", "S064593", "S066081", "S066995", "S067212", "S068549", "S069165", "S060874", "S061590", "S061768", "S062470", "S062525", "S062673", "S063107", "S063978", "S064115", "S064380", "S064576", "S064706", "S064779", "S067773", "S060982", "S061161", "S061171", "S061345", "S061984", "S062351", "S062534", "S062947", "S062961", "S063017", "S063667", "S063805", "S064244", "S064246", "S064444", "S064472", "S065090", "S066416", "S068351"]
#vinList = ["S061174"]
out_feat_file = open("feature_extracted_test.txt", "w")

for vin in vinList:
    file = "../DataLoader/sortedData/"+vin+".dat"
    with open(file,'r') as f:
        allSeries = {}
        for line in f: #read in record
            tuples = line.split(",")
            vin = tuples[1]
            timeInfo = parser.parse(tuples[2])

            date = str(timeInfo.year)+"-"+str(timeInfo.month)+"-"+str(timeInfo.day)
            time = str(timeInfo.hour)+":"+str(timeInfo.minute)+":"+str(timeInfo.second)
            b_soc = tuples[14]
            b_volt = tuples[15]
            c_hv = tuples[18]
            c_lv = tuples[21]
            lat = tuples[80]
            lng = tuples[81]
            odo = tuples[85]
            orientation = tuples[86]
            speed = tuples[161]
            #print b_soc,b_volt
            fuel = tuples[165].strip('\n')

            a = {"vin": vin, "date":date, "time":time, "b_soc":b_soc, "b_volt":b_volt, "c_hv": c_hv, "c_lv":c_lv, "lat": lat, "lng": lng, "odo": odo, "orientation":orientation, "speed": speed, "fuel":fuel}

            if(date not in allSeries):
                allSeries[date] = {}
                allSeries[date]['rawData'] = []
            allSeries[date]['rawData'].append(a)

        for date,v in allSeries.iteritems():    #for each day
            allData = v['rawData']              #all series data sorted by time

            #odometer information
            odo = filter(lambda x:x!='\\N',map(lambda x:x['odo'],v['rawData']))

            if(len(odo)!=0):
                odo = reject_outliers(map(lambda x:float(x),odo),1000)
                v['daily_distance'] = odo[-1]-odo[0]
            else:
                v['daily_distance'] = 0
            if(v['daily_distance']<0):
                v['daily_distance'] = 0

            #position info with time {lat,lng,time}
            position_raw = map(lambda x:{'lat':float(x['lat']),'lng':float(x['lng']),'time':x['time']},v['rawData'])
            position = filter(lambda x:x['lat']>25 and x['lat']<40 and x['lng']>115 and x['lng']<130,position_raw)

            lat = map(lambda x:x['lat'],position)
            latmean = np.mean(lat)
            latstd = np.std(lat)
            lng = map(lambda x:x['lng'],position)
            lngmean = np.mean(lng)
            lngstd = np.std(lng)
            pos_processed = []
            for idx,val in enumerate(position):
                if(len(pos_processed)==0):
                    if(abs(val['lat']-latmean)<max(1,latstd) and abs(val['lng']-lngmean)<max(1,lngstd)):
                        pos_processed.append(val)
                else:
                    if(abs(val['lat']-pos_processed[len(pos_processed)-1]['lat'])<1 and abs(val['lng']-pos_processed[len(pos_processed)-1]['lng'])<1):
                        pos_processed.append(val)
            v['position'] = pos_processed

            #fuel info
            fuel = filter(lambda x:x!='none',map(lambda x: 'none' if x['fuel']=='\\N' else float(x['fuel']) ,v['rawData']))
            v['fuel'] = np.mean(fuel)

            #charger
            charger = map(lambda x:{'hv':x['c_hv'],'lv':x['c_lv'],'time':x['time']},v['rawData'])
            chgRecord = []
            startCount = False
            chgtmp = []
            for c,p in zip(charger,allData):
                if(c['hv']!='\\N'): #in charging
                    if(startCount==False):
                        chgtmp = []
                        startCount = True
                        chgtmp.append({"time":c['time'],"position":{"lat":p['lat'],"lng":p['lng']}})
                    else:
                        prevLat = chgtmp[-1]["position"]["lat"]
                        prevLng = chgtmp[-1]["position"]["lng"]
                        if(float(prevLat)-float(p['lat'])<0.001 and float(prevLng)-float(p['lng'])<0.001):
                            #0.001 lat and lng in 30 degree North (shanghai) = 9 meters
                            chgtmp.append({"time":c['time'],"position":{"lat":p['lat'],"lng":p['lng']}})
                else:
                    if(startCount==True):
                        duration = parser.parse(chgtmp[-1]["time"])-parser.parse(chgtmp[0]["time"])
                        lat = np.mean(map(lambda x: float(x["position"]["lat"]),chgtmp))
                        lng = np.mean(map(lambda x: float(x["position"]["lng"]),chgtmp))
                        startCount = False
                        if(duration.seconds > 60):
                            chgRecord.append({"duration_seconds":duration.seconds,"lat":lat,"lng":lng})
            v['charger'] = chgRecord

            #stop positions
            stopRecord = []
            startCount = False
            stoptmp = []
            for i in allData:
                speed = i['speed']
                lat = i['lat']
                lng = i['lng']
                time = i['time']
                if(i['speed']=='0.0' or i['speed']=='\N'):
                    if(startCount==False):
                        stoptmp = []
                        startCount = True
                        stoptmp.append({"lat":lat,"lng":lng,"time":time})
                    else:
                        prevLat = stoptmp[-1]["lat"]
                        prevLng = stoptmp[-1]["lng"]
                        if(float(prevLat)-float(lat)<0.01 and float(prevLng)-float(lng)<0.01):
                            #0.001 lat and lng in 30 degree North (shanghai) = 9 meters
                            stoptmp.append({"lat":lat,"lng":lng,"time":time})
                else:
                    if(startCount == True):
                        duration = (parser.parse(stoptmp[-1]["time"])-parser.parse(stoptmp[0]["time"])).seconds
                        startCount = False
                        if(duration>100):
                            lat = np.mean(map(lambda x: float(x["lat"]),stoptmp))
                            lng = np.mean(map(lambda x: float(x["lng"]),stoptmp))
                            stopRecord.append({"duration_seconds":duration,"lat":lat,"lng":lng})
            v['stops'] = stopRecord

            #batteryInfo
            btmp = []
            for i in allData:
                speed = i["speed"]
                c = i["c_hv"]
                if(speed !='0.0' and speed !='\N' and c=='\\N'):
                    btmp.append({"time":i["time"],"b_soc":i["b_soc"],"b_volt":i["b_volt"]})
            BatRecord = []
            dailyRecord = []
            tmp = []
            for j in btmp:
                if(len(tmp)==0):
                    tmp.append(j)
                else:
                    if((parser.parse(j['time'])-parser.parse(tmp[-1]["time"])).seconds>100):#stop for more than 100 second, split
                        if((parser.parse(tmp[-1]['time'])-parser.parse(tmp[0]['time'])).seconds>600): # timespan> 600 second, recorded, else discarded,
                            dailyRecord.append(tmp)
                            tmp = [j]
                        else:
                            tmp = [j]
                    else:
                        tmp.append(j)
            for records in dailyRecord:
                timespan = (parser.parse(records[-1]["time"])-parser.parse(records[0]["time"])).seconds
                b_soc = map(lambda x:x["b_soc"],records)
                b_volt = map(lambda x:x["b_volt"],records)
                records = {"timespan":timespan,"b_soc_max":max(b_soc),"b_soc_min":min(b_soc),"b_soc_first":b_soc[0],"b_soc_last":b_soc[-1],"b_volt_max":max(b_volt),"b_volt_min":min(b_volt),"b_volt_first":b_volt[0],"b_volt_last":b_volt[-1],"b_soc":b_soc,"b_volt":b_volt}
                BatRecord.append(records)
            v['battery'] = BatRecord
        '''
        for date,v in allSeries.iteritems():
            print date
            print "daily distance:",v["daily_distance"]
            print "position track:",v["position"]
        '''
        holiday = ["2016-4-30","2016-5-1","2016-5-2","2016-5-7","2016-5-8"]
        workday = ["2016-4-29","2016-5-3","2016-5-4","2016-5-5","2016-5-6","2016-5-9","2016-5-10","2016-5-11"]

        holidayRecord = [allSeries[k] for k in set(holiday).intersection(set(allSeries.keys()))]
        holidayRecord = filter(lambda x:x["daily_distance"]!=0,holidayRecord)
        workdayRecord = [allSeries[k] for k in set(workday).intersection(set(allSeries.keys()))]
        workdayRecord = filter(lambda x:x["daily_distance"]!=0,workdayRecord)
        all_keys = list(set(holiday).union(set(workday)).intersection(allSeries.keys()))
        allRecord = [allSeries[k] for k in all_keys]

        # # of records in holiday (where distance != 0)
        feat_Num_Records_Holiday = len(holidayRecord)
        # # of records in workday (where distance != 0)
        feat_Num_Records_Workday = len(workdayRecord)
        # mean distance in holiday
        feat_Mean_Distance_Holiday = np.nanmean(map(lambda x:x["daily_distance"],holidayRecord))
        if(np.isnan(feat_Mean_Distance_Holiday)):
            feat_Mean_Distance_Holiday = 0
        #mean distance in workday
        feat_Mean_Distance_Workday = np.nanmean(map(lambda x:x["daily_distance"],workdayRecord))
        if(np.isnan(feat_Mean_Distance_Workday)):
            feat_Mean_Distance_Workday = 0
        #variance distance in workday
        feat_Var_Distance_Workday = np.nanstd(map(lambda x:x["daily_distance"],workdayRecord))
        if(np.isnan(feat_Var_Distance_Workday)):
            feat_Var_Distance_Workday = 0
        #mean fuel everyday
        mean_fuel = np.nanmean(map(lambda x:x['fuel'],allRecord))
        if(np.isnan(mean_fuel)):
            mean_fuel = 6.5
        #variance fuel everyday
        feat_Var_Fuel_All = np.nanstd(map(lambda x:x['fuel'],allRecord))
        if(np.isnan(feat_Var_Fuel_All)):
            feat_Var_Fuel_All = 0
        #mean time of charges in holiday
        feat_Mean_ChgTime_Holiday = np.nanmean(map(lambda x:np.nansum(map(lambda y:y['duration_seconds'],x['charger'])),holidayRecord))
        if(np.isnan(feat_Mean_ChgTime_Holiday)):
            feat_Mean_ChgTime_Holiday = 0
        #mean time of charges in workday
        feat_Mean_ChgTime_Workday = np.nanmean(map(lambda x:np.nansum(map(lambda y:y['duration_seconds'],x['charger'])),workdayRecord))
        if(np.isnan(feat_Mean_ChgTime_Workday)):
            feat_Mean_ChgTime_Workday = 0
        ## of stops > 10min in holiday
        feat_Num_LongStop_Holiday = np.nanmean(map(lambda x:np.nansum(map(lambda y:0 if(y['duration_seconds']<600)else 1,x['stops'])),holidayRecord))
        if(np.isnan(feat_Num_LongStop_Holiday)):
            feat_Num_LongStop_Holiday = 0
        ## of stops > 10min in workday
        feat_Num_LongStop_Workday =np.nanmean(map(lambda x:np.nansum(map(lambda y:0 if(y['duration_seconds']<600)else 1,x['stops'])),workdayRecord))
        if(np.isnan(feat_Num_LongStop_Holiday)):
            feat_Num_LongStop_Holiday = 0
        ## of stops < 10min in holiday
        feat_Num_ShortStop_Holiday =np.nanmean(map(lambda x:np.nansum(map(lambda y:1 if(y['duration_seconds']<600)else 0,x['stops'])),holidayRecord))
        if(np.isnan(feat_Num_ShortStop_Holiday)):
            feat_Num_ShortStop_Holiday = 0
        ## of stops < 10min in workday
        feat_Num_ShortStop_Workday =np.nanmean(map(lambda x:np.nansum(map(lambda y:1 if(y['duration_seconds']<600)else 0,x['stops'])),workdayRecord))
        if(np.isnan(feat_Num_ShortStop_Workday)):
            feat_Num_ShortStop_Workday = 0

        #max horizontal lng span in holiday
        holiday_horizontal_list = map(lambda x:map(lambda y:y['lng'],x['position']),holidayRecord)
        holiday_horizontal = [records for day in holiday_horizontal_list for records in day]
        if(len(holiday_horizontal)==0):
            feat_Max_HorSpan_Holiday = 0
        else:
            feat_Max_HorSpan_Holiday = np.nanmax(holiday_horizontal)-np.nanmin(holiday_horizontal)

        #max vertical lat span in holiday
        holiday_vertical_list = map(lambda x:map(lambda y:y['lat'],x['position']),holidayRecord)
        holiday_vertical = [records for day in holiday_vertical_list for records in day]
        if(len(holiday_vertical)==0):
            feat_Max_VertSpan_Holiday = 0
        else:
            feat_Max_VertSpan_Holiday = np.nanmax(holiday_vertical)-np.nanmin(holiday_vertical)

        #max horizontal lng span in workday
        workday_horizontal_list = map(lambda x:map(lambda y:y['lng'],x['position']),workdayRecord)
        workday_horizontal = [records for day in workday_horizontal_list for records in day]
        if(len(workday_horizontal)==0):
            feat_Max_HorSpan_Workday = 0
        else:
            feat_Max_HorSpan_Workday = np.nanmax(workday_horizontal)-np.nanmin(workday_horizontal)

        #max vertical lat span in workday
        workday_vertical_list = map(lambda x:map(lambda y:y['lat'],x['position']),workdayRecord)
        workday_vertical = [records for day in workday_vertical_list for records in day]
        if(len(workday_vertical)==0):
            feat_Max_VertSpan_Workday = 0
        else:
            feat_Max_VertSpan_Workday = np.nanmax(workday_vertical)-np.nanmin(workday_vertical)
        #largest horizontal span in workday within 2(or 1~1.5, more strict) sigma space of Normal distribution (97.7% confidence interval (or 68%))
        workday_conf_hor_list = map(lambda x:reject_outliers_mean(map(lambda y:y['lng'],x['position']),2),workdayRecord)
        workday_conf_hor = [records for day in workday_conf_hor_list for records in day]
        feat_Conf_HorSpan_Workday = np.nanmax(workday_conf_hor)-np.nanmin(workday_conf_hor)
        left_extrema = np.nanmin(workday_conf_hor)
        right_extrema = np.nanmax(workday_conf_hor)
        #largest vertical span in workday within 2(or 1~1.5 more strict) sigma space of Normal distribution (97.7% confidence interval (or 68%))
        workday_conf_ver_list = map(lambda x:reject_outliers_mean(map(lambda y:y['lat'],x['position']),2),workdayRecord)
        workday_conf_ver = [records for day in workday_conf_ver_list for records in day]
        feat_Conf_VerSpan_Workday = np.nanmax(workday_conf_ver)-np.nanmin(workday_conf_ver)
        top_extrema = np.nanmax(workday_conf_ver)
        btm_extrema = np.nanmin(workday_conf_ver)

        features = [feat_Num_Records_Holiday,feat_Num_Records_Workday,feat_Mean_Distance_Holiday,feat_Mean_Distance_Workday,feat_Var_Distance_Workday,mean_fuel,feat_Var_Fuel_All,feat_Mean_ChgTime_Holiday,
                    feat_Mean_ChgTime_Workday,feat_Num_LongStop_Holiday,feat_Num_LongStop_Workday,feat_Num_ShortStop_Holiday,feat_Num_ShortStop_Workday,feat_Max_HorSpan_Holiday,
                    feat_Max_VertSpan_Holiday,feat_Max_HorSpan_Workday,feat_Max_VertSpan_Workday,feat_Conf_HorSpan_Workday,feat_Conf_VerSpan_Workday]
        #matrix generation
        mat_dim = 24
        u_Hor = feat_Conf_HorSpan_Workday/mat_dim
        u_Ver = feat_Conf_VerSpan_Workday/mat_dim
        for idx,dayRecord in enumerate(allRecord):
            position_matrix = np.zeros(shape=(mat_dim,mat_dim))
            #positions
            position_record = dayRecord['position']
            last_lng = -1
            last_lat = -1
            c=0
            for point in position_record:
                d_lng = int((point['lng']-left_extrema)/u_Hor)
                d_lat = int((point['lat']-btm_extrema)/u_Ver)
                if(d_lng != last_lng and d_lat != last_lat and d_lng<mat_dim and d_lat<mat_dim and d_lng>-1 and d_lat>-1):
                    position_matrix[d_lat][d_lng]+=0.2
                    last_lat = d_lat
                    last_lng = d_lng
                    if(position_matrix[d_lat][d_lng]>1):
                        position_matrix[d_lat][d_lng]=1
            #stops
            stop_record = dayRecord['stops']
            for point in stop_record:
                d_lng = int((point['lng']-left_extrema)/u_Hor)
                d_lat = int((point['lat']-btm_extrema)/u_Ver)
                if(d_lng<mat_dim and d_lat<mat_dim and d_lng>-1 and d_lat>-1):
                    if(point["duration_seconds"]<600):
                        position_matrix[d_lat][d_lng] += 0.4
                    else:
                        position_matrix[d_lat][d_lng] += 0.6
                    if(position_matrix[d_lat][d_lng]>1):
                        position_matrix[d_lat][d_lng]=1
            #chargings
            chg_record = dayRecord['charger']
            if(len(chg_record)!=0):
                for point in chg_record:
                    d_lng = int((point['lng']-left_extrema)/u_Hor)
                    d_lat = int((point['lat']-btm_extrema)/u_Ver)
                    if(d_lng<mat_dim and d_lat<mat_dim and d_lng>-1 and d_lat>-1):
                        if(point["duration_seconds"]<600):
                            position_matrix[d_lat][d_lng] += 0.4
                        else:
                            position_matrix[d_lat][d_lng] += 0.6
                        if(position_matrix[d_lat][d_lng]>1):
                            position_matrix[d_lat][d_lng]=1
            plt.gray()
            plt.imsave("img/"+vin+"at"+all_keys[idx]+".jpg",position_matrix)
            filename = "matrix/"+vin+"."+all_keys[idx]+".mat"
            np.savetxt(filename,position_matrix,fmt='%10.2f')

        features = map(lambda x:str(x),features)
        out_feat_file.write(vin + ":")
        out_feat_file.write(",".join(features))
        out_feat_file.write("\n")
'''
for date,v in allSeries.iteritems():
            print date
            print "daily distance:",v["daily_distance"]
            # print "position track:",v["position"]
            print "avg fuel",v["fuel"]
            print "charging record:",len(v["charger"]),v["charger"]
            print "stop record:",len(v["stops"]),v["stops"]
            print "battery discharge record",v["battery"]
            print "=============date end=============="
'''


out_feat_file.close()

'''
Explain:

these are all records for each day
All data structures are list of dictionaries, each dictionary is related tuple/statistics for original data in one row of sorted data

daily_distance: distance this day , from odometer
position:       gps [lat,lng,time]
avg fuel:       most of the cases, car fuel consumption is 5~12. When car use electricity, the fuel=0. On avg, electricity or hybrid power < fuel in terms of this
charger:        charging record[lat,lng,duration]
stops:          stops that duration > 100 seconds
battery:        split the discharge records of battery into several splits that: 1)each split contains data > 600s. 2)interval between splits > 100s
                data format:[timespan, b_soc_first/last/max/min, b_volt_first/last/max/min, b_soc, b_volt]

'''