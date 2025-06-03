import math
import sys
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")


"""Feeder functions"""

def neighborhood(iterable):
    iterator = iter(iterable)
    prev = (0)
    item = next(iterator)  # throws StopIteration if empty.
    for n in iterator:
        yield (prev,item,n)
        prev = item
        item = n
    yield (prev,item,None)

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out

"""Non-feeder functions"""

def get_pkt_list(trace_data):
    first_line = trace_data[0]
    first_line = first_line.split("\t")

    first_time = float(first_line[0])
    dta = []
    for line in trace_data:
        a = line
        b = a.split("\t")

        if float(b[1]) > 0:
            #dta.append(((float(b[0])- first_time), abs(int(b[2])), 1))
            dta.append(((float(b[0])- first_time), 1))
        else:
            #dta.append(((float(b[1]) - first_time), abs(int(b[2])), -1))
            dta.append(((float(b[0]) - first_time), -1))
    return dta


def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] < 0:
            In.append(p)
        if p[1] > 0:
            Out.append(p)
    return In, Out

############### TIME FEATURES #####################

def inter_pkt_time(list_data):
    # times = [x[0] for x in list_data]
    # temp = []
    # for elem,next_elem in zip(times, times[1:]+[times[0]]):
    #     temp.append(next_elem-elem)
    # return temp[:-1]

    temp = []
    for i in range(len(list_data) - 1):
        temp.append(list_data[i + 1][0] - list_data[i][0])
    return temp
    

def interarrival_times(inout):
    In, Out = inout
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    # TOTAL = inter_pkt_time(list_data)
    TOTAL = IN + OUT
    return IN, OUT, TOTAL

def interarrival_maxminmeansd_stats(inout):
    In, Out, Total = interarrival_times(inout)
    if In and Out:
        avg_in = sum(In)/float(len(In))
        avg_out = sum(Out)/float(len(Out))
        avg_total = sum(Total)/float(len(Total))
        interstats = (max(In), max(Out), max(Total), avg_in, avg_out, avg_total, np.std(In), np.std(Out), np.std(Total), np.percentile(In, 75), np.percentile(Out, 75), np.percentile(Total, 75))
    elif Out and not In:
        avg_out = sum(Out)/float(len(Out))
        avg_total = sum(Total)/float(len(Total))
        interstats = (0, max(Out), max(Total), 0, avg_out, avg_total, 0, np.std(Out), np.std(Total), 0, np.percentile(Out, 75), np.percentile(Total, 75))
    elif In and not Out:
        avg_in = sum(In)/float(len(In))
        avg_total = sum(Total)/float(len(Total))
        interstats = (max(In), 0, max(Total), avg_in, 0, avg_total, np.std(In), 0, np.std(Total), np.percentile(In, 75), 0, np.percentile(Total, 75))
    else:
        interstats = [0] * 12
    return interstats

def time_percentile_stats(Total, inout):
    In, Out = inout
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25)) # return 25th percentile
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend(([0]*4))
    if Out1:
        STATS.append(np.percentile(Out1, 25)) # return 25th percentile
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend(([0]*4))
    if Total1:
        STATS.append(np.percentile(Total1, 25)) # return 25th percentile
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend(([0]*4))
    return STATS

def number_pkt_stats(inout):
    In, Out = inout
    return len(In), len(Out), len(In) + len(Out)

def first_and_last_30_pkts_stats(inout):
    In, Out = inout
    first30in = In[:30]
    first30out = Out[:30]
    last30in = In[-30:]
    last30out = Out[-30:]
    stats= []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats

#concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(Total):
    chunks= [Total[x:x+20] for x in range(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] > 0:
                c+=1
        concentrations.append(c)
    return np.std(concentrations), sum(concentrations)/float(len(concentrations)), np.percentile(concentrations, 50), min(concentrations), max(concentrations), concentrations

#Average number packets sent and received per second
def number_per_sec(Total):
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    
    # temp = []
    # l = []
    # for i in range(1, int(last_second)+1):
    #     c = 0
    #     for p in Total:
    #         if p[0] <= i:
    #             c+=1
    #     temp.append(c)
    # for prev,item,next in neighborhood(temp):
    #     x = item - prev
    #     l.append(x)
    
    l = []
    ptr = 0
    for sec in range(1, int(last_second)+1):
        cnt = 0
        while ptr < len(Total) and Total[ptr][0] <= sec:
            cnt += 1
            ptr += 1
        l.append(cnt)

    if len(l) == 0:
        return 0, 0, 0, 0, 0, []
    else: 
        return sum(l)/float(len(l)), np.std(l), np.percentile(l, 50), min(l), max(l), l

#Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(Total):
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] > 0:
            temp1.append(c1)
        c1+=1
        if p[1] < 0:
            temp2.append(c2)
        c2+=1
    avg_in = sum(temp1)/float(len(temp1)) if len(temp1) > 0 else 0
    avg_out = sum(temp2)/float(len(temp2)) if len(temp2) > 0 else 0

    return avg_in, avg_out, np.std(temp1), np.std(temp2)

def perc_inc_out(inout):
    In, Out = inout
    len_total = len(In) + len(Out)
    percentage_in = len(In)/float(len_total)
    percentage_out = len(Out)/float(len_total)
    return percentage_in, percentage_out

############### SIZE FEATURES #####################

def total_size(length_list):
   return sum(length_list)

def in_out_size(inout):
   In, Out = inout
   size_in = sum([x[1] for x in In])
   size_out = sum([x[1] for x in Out])
   return size_in, size_out

def average_total_pkt_size(length_list):
   return np.mean(length_list)

def average_in_out_pkt_size(inout):
   In, Out = inout
   average_size_in = np.mean([x[1] for x in In])
   average_size_out = np.mean([x[1] for x in Out])
   return average_size_in, average_size_out

def variance_total_pkt_size(length_list):
   return np.var(length_list)

def variance_in_out_pkt_size(inout):
   In, Out = inout
   var_size_in = np.var([x[1] for x in In])
   var_size_out = np.var([x[1] for x in Out])
   return var_size_in, var_size_out

def std_total_pkt_size(length_list):
   return np.std(length_list)

def std_in_out_pkt_size(inout):
   In, Out = inout
   std_size_in = np.std([x[1] for x in In])
   std_size_out = np.std([x[1] for x in Out])
   return std_size_in, std_size_out

def max_in_out_pkt_size(inout):
   In, Out = inout
   max_size_in = np.max([x[1] for x in In]) if len(In) > 0 else 0
   max_size_out = np.max([x[1] for x in Out]) if len(Out) > 0 else 0
   return max_size_in, max_size_out


############### FEATURE FUNCTION #####################


#If size information available add them in to function below
def TOTAL_FEATURES(cell, max_size=175, no_length=False):
    ALL_FEATURES = []
    
    inout = In_Out(cell)
    length_list = [x[1] for x in cell]
    
    # ------TIME--------
    if len(inout[0]) == 0:
        cell.append((cell[-1][0], 1))
        inout = In_Out(cell)
    if len(inout[1]) == 0:
        cell.append((cell[-1][0], -1))
        inout = In_Out(cell)
    intertimestats = interarrival_maxminmeansd_stats(inout)
    timestats = time_percentile_stats(cell, inout)
    number_pkts = number_pkt_stats(inout)
    thirtypkts = first_and_last_30_pkts_stats(inout)
    stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(cell)
    avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = number_per_sec(cell)
    avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(cell)
    perc_in, perc_out = perc_inc_out(inout)

    altconc = []
    alt_per_sec = []
    altconc = [sum(x) for x in chunkIt(conc, 70)]
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    if len(altconc) == 70:
        altconc.append(0)
    if len(alt_per_sec) == 20:
        alt_per_sec.append(0)

    # ------SIZE--------
    if not no_length:
        tot_size = total_size(length_list)
        in_size, out_size = in_out_size(inout)
        avg_total_size = average_total_pkt_size(length_list)
        avg_size_in, avg_size_out = average_in_out_pkt_size(inout)
        var_total_size = variance_total_pkt_size(length_list)
        var_size_in, var_size_out = variance_in_out_pkt_size(inout)
        std_total_size = std_total_pkt_size(length_list)
        std_size_in, std_size_out = std_in_out_pkt_size(inout)
        max_size_in, max_size_out = max_in_out_pkt_size(inout)

        # SIZE FEATURES
        ALL_FEATURES.append(tot_size)
        ALL_FEATURES.append(in_size)
        ALL_FEATURES.append(out_size)
        ALL_FEATURES.append(avg_total_size)
        ALL_FEATURES.append(avg_size_in)
        ALL_FEATURES.append(avg_size_out)
        ALL_FEATURES.append(var_total_size)
        ALL_FEATURES.append(var_size_in)
        ALL_FEATURES.append(var_size_out)
        ALL_FEATURES.append(std_total_size)
        ALL_FEATURES.append(std_size_in)
        ALL_FEATURES.append(std_size_out)
        ALL_FEATURES.append(max_size_in)
        ALL_FEATURES.append(max_size_out)

    # TIME Features
    ALL_FEATURES.extend(intertimestats)
    ALL_FEATURES.extend(timestats)
    ALL_FEATURES.extend(number_pkts)
    ALL_FEATURES.extend(thirtypkts)
    ALL_FEATURES.append(stdconc)
    ALL_FEATURES.append(avgconc)
    ALL_FEATURES.append(avg_per_sec)
    ALL_FEATURES.append(std_per_sec)
    ALL_FEATURES.append(avg_order_in)
    ALL_FEATURES.append(avg_order_out)
    ALL_FEATURES.append(std_order_in)
    ALL_FEATURES.append(std_order_out)
    ALL_FEATURES.append(medconc)
    ALL_FEATURES.append(med_per_sec)
    ALL_FEATURES.append(min_per_sec)
    ALL_FEATURES.append(max_per_sec)
    ALL_FEATURES.append(maxconc)
    ALL_FEATURES.append(perc_in)
    ALL_FEATURES.append(perc_out)
    ALL_FEATURES.extend(altconc)
    ALL_FEATURES.extend(alt_per_sec)
    ALL_FEATURES.append(sum(altconc))
    ALL_FEATURES.append(sum(alt_per_sec))
    ALL_FEATURES.append(sum(intertimestats))
    ALL_FEATURES.append(sum(timestats))
    ALL_FEATURES.append(sum(number_pkts))


    # This is optional, since all other features are of equal size this gives the first n features
    # of this particular feature subset, some may be padded with 0's if too short.

    ALL_FEATURES.extend(conc)

    ALL_FEATURES.extend(per_sec)


    while len(ALL_FEATURES)<max_size:
        ALL_FEATURES.append(0)
    features = ALL_FEATURES[:max_size]
    
    # replace NaN with 0
    imp = SimpleImputer(strategy="constant", fill_value=0)
    feats = imp.fit_transform([features])[0].tolist()
    
    return feats
