import time
import datetime
import pytz
#
time = int(time.time())
print(time)
utc  = datetime.datetime.utcfromtimestamp(time)
print(utc)
time_list = str(utc).split(" ")
utc_time = time_list[0] + "T" + time_list[1] + "Z"
print(utc_time)

# utc_now = datetime.datetime.strptime(str(utc),'%Y-%m-%d %H:%M:%S')
# print(utc_now)
#
# print(utc)
# tz  = pytz.timezone('UTC')
# now = datetime.datetime.now(tz)
# print(now)
# str_now = now.strptime("%Y-%m-%dT%H:%M:%S.%f%z")
# print(str_now)
# tim = "2019-08-22T00:07:00Z"
# utc_date = datetime.datetime.strptime(tim, "%Y-%m-%dT%H:%M:%SZ")
# local_date = utc_date + datetime.timedelta(hours=8)
# local_date_str = datetime.datetime.strptime(str(local_date), "%Y-%m-%d %H:%M:%S")
# t = local_date_str.timestamp()
#
# print(int(t))
# now = datetime.datetime.fromtimestamp(t)
# print(now)

a = [
    (1,2,3),(4,5,6),(7,8,9)
]
for i in a:
    print(i[0])