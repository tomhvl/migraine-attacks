# 03/01/2016 changed attribute naming (shorter)
#            modified flag output to 1:0 from T/False
#            adapted dateformat to new updated lstamp version

# 28/01/2016 anomaly count report will now print total number of rows instead of startrow
#            startrow is still used, but not printed in the report

# 05/02/2017 agg_value was not passed to model even when aggregate steps were activated!!
#            activity_value was used (bad bug)



import sys, os.path, csv
import argparse
import pandas as pd
import itertools

import migraine_processing as mig


from datetime import datetime

from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.algorithms import anomaly_likelihood
from nupic.data.inference_shifter import InferenceShifter

# assume data has been swarmed and model created, date format used in datafile
DATE_FORMAT = "%Y-%m-%d %H:%M"

# allowed distance (in timesteps) from detected anomaly to actual labelled anomaly ("M" annotation)
# population could then be defined as the total number of data points / period length ??
# for the purpose off calculating accuracy/precision/recall
OVERLAP_PERIOD = 120

# if this is =1 then values are used as is, otherwise they are divided by this value
# with 1 minute epochs, values / 60  -> average 1Hz readings (per second)
ADJUST_POINT_FACTOR = 1


ANOMALY_COUNT_REPORT_OUTFILE = "anomaly_count_stats.csv"

# change directory below for other models
#from model_params import mp_h1_d0_w1_a05_n400_rad30_bst1_pam1 as model_params
# mp_h0_d0_w0_a02_n250_bst1_pam1_oldswarm_D
#from model_params import mp_h0_d0_w0_a008_n238_rad0_bst1_pam2_newdateswm_D_cut as model_params
#from model_params import mp_h0_d0_w0_a038_n1000_rad9_bsmetric_D as model_params
from model_params import mp_h0_d0_w0_a05_n200_D as model_params

modelParams = model_params.MODEL_PARAMS


def setActiveModelParams(alpha, maxboost, value, time, pam):
    #model = ModelFactory.create(modelParams)

    #print model_params.MODEL_PARAMS['modelParams']['clParams']['alpha']
    #modelParams['modelParams']['clParams']['alpha'] = alpha
    modelParams['modelParams']['spParams']['maxBoost'] = maxboost
    modelParams['modelParams']['sensorParams']['encoders']['value']['n'] = value
    modelParams['modelParams']['sensorParams']['encoders']['timestamp_weekend'] = time
    modelParams['modelParams']['tpParams']['pamLength'] = pam


    model = ModelFactory.create(modelParams)
    # print model.getFieldInfo()
    # print model.getInferenceArgs()
    # print model.getRuntimeStats()
    #print model_params.MODEL_PARAMS['modelParams']['clParams']['alpha']

    return model

# return a pandas dataframe
def getData(filename):
    return mig.getCaseData(filename)

# main method, start is the starting row from which actual anomaly statistics are counted
# reason for this is a certain number of records/rows is needed to learn the sequence
def run_experiment(model, filename, startrow, aggstep, aggfunction):

    model.enableInference({"predictedField": "value"})



    df_in = getData(filename)#mig.getCaseData("stm_sinewave_anomaly.csv") #getData(filename)
    #bad_data = df_in.value.rolling(180, center=True).sum() == 0
    #df_in = df_in.iloc[bad_data[0]:]
    outdata = []
    outdata_index = []

    counter = 0

    # # temporary counters to help finding overlap between anomalyscore & registered migraine attacks
    # flagged_counter = 0
    # raw_anomaly_counter = 0
    # likelihood_counter = 0
    # loglhood_counter = 0

    # anomalyLikelihood object for likelihood prob calculcation
    anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood()

    # shifter = InferenceShifter()

    step = 0
    agg_value = 0
    last_value = 0
    step_flag = False

    window = [0]*aggstep

    for row in df_in.itertuples(name=None):

        timestamp = row[0].to_pydatetime()
        activity_value = row[1] * ADJUST_POINT_FACTOR
        agg_value += activity_value
        flagged = row[2] or step_flag

        # update sliding window "que"
        #window, aggregate = slideWindow(window, activity_value)

        # when using delta, calculate:
        delta = agg_value - last_value
        last_value = agg_value
        agg_value = delta
        # print delta


        step += 1
        # update on progress into datafile
        counter+=1
        if counter%1000 == 0:
            print counter,"rows processed"

        if (step%aggstep!=0):
            step_flag = flagged
            continue

        #print (agg_value)
        # forget step_flag value , step and call aggregate function sum=default, else mean
        if (aggfunction != "sum"):
            agg_value = agg_value/aggstep

        step_flag = False
        step = 0

        # run the model on the data
        result = model.run({
               "timestamp": timestamp,
               "value": agg_value
        })

        anomalyScore = result.inferences["anomalyScore"]
        likelihood = anomalyLikelihood.anomalyProbability(agg_value, anomalyScore, timestamp)
        logLikelihood = anomalyLikelihood.computeLogLikelihood(likelihood)

        #result = shifter.shift(result)
        #lastPrediction = prediction
        #lastPrediction = result.inferences["multiStepBestPredictions"][1]
        lastPrediction = -1.0





        outdata_index.append(timestamp)
        outdata.append([agg_value, flagged, anomalyScore, round(likelihood,5), \
                round(logLikelihood, 2), round(lastPrediction, 2)])

        agg_value = 0


    # output dataframe to csv file
    df_out = pd.DataFrame(outdata, columns=["value", "flag", "a_score", \
                    "likelihood", "log_lhood", "pred"], index=outdata_index)
    df_out.to_csv("APD_acvout_"+filename)

    # calculate metrics

    #final_result = [raw_overlap, lkh_overlap, log_overlap, flag_total, flag_after_start, counter, -1]

    final_result = calculateMetrics(df_out, startrow)
    final_result.insert(0, filename)

    print final_result

    return final_result


def calculateMetrics(df, startrow):
    # calc recall based on dataframe
    # thresholds
    AS = 1.0; LH = 0.9999; LG = 0.5

    before = mig.getFlaggedSeqList(df.iloc[startrow:], pre=OVERLAP_PERIOD, post=0)
    after = mig.getFlaggedSeqList(df.iloc[startrow:], pre=0, post=OVERLAP_PERIOD)

    raw_overlap = [0,0,0]
    lkh_overlap = [0,0,0]
    log_overlap = [0,0,0]
    recall = [0,0,0]
    flag_total = df.flag[df.flag].count()
    flag_after_start = df.flag.iloc[startrow:][df.flag].count()
    counter = df.flag.size

    # for each interval, get count of each score metric
    for (fra, til) in before:
        raw_overlap[0] += (df.a_score[fra:til][df.a_score==AS]).count()
        lkh_overlap[0] += (df.likelihood[fra:til][df.likelihood>LH]).count()
        log_overlap[0] += (df.log_lhood[fra:til][df.log_lhood>LG]).count()

    for (fra, til) in after:
        traw = raw_overlap[1]
        tlkh = lkh_overlap[1]
        tlog = log_overlap[1]
        raw_overlap[1] += df.a_score[fra:til][df.a_score==AS].count()
        lkh_overlap[1] += df.likelihood[fra:til][df.likelihood>LH].count()
        log_overlap[1] += df.log_lhood[fra:til][df.log_lhood>LG].count()
        recall[0] += int(raw_overlap[1] > traw)
        recall[1] += int(lkh_overlap[1] > tlkh)
        recall[2] += int(log_overlap[1] > tlog)

    raw_overlap[2] = df.a_score[startrow:][df.a_score==AS].count()
    lkh_overlap[2] = df.likelihood[startrow:][df.likelihood>LH].count()
    log_overlap[2] = df.log_lhood[startrow:][df.log_lhood>LG].count()

    #df = df.drop('s', axis=1)
    return [raw_overlap, lkh_overlap, log_overlap, flag_total, flag_after_start, counter, recall]




# update sliding window and aggregate (sum) value of it
def slideWindow(window, last_value):
    aggregate = last_value
    window[0] = last_value

    if len(window) > 1:
        for i in range(1,len(window)):
            window[i-1] = window[i]
            aggregate += window[i]
        window[i]=last_value

    return window, aggregate

def makeLogEntry(current_stats):
    # output the anomaly statistics to seperate file
    countfile = csv.writer(open(ANOMALY_COUNT_REPORT_OUTFILE, "a"))
    datestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    current_stats.insert(0, datestamp)
    countfile.writerow(current_stats)

def makeLogHeader(header_info):
    exist_countfile = os.path.isfile(ANOMALY_COUNT_REPORT_OUTFILE)
    countfile = csv.writer(open(ANOMALY_COUNT_REPORT_OUTFILE, "a"))

    # only write this 1st time
    if not exist_countfile:
        countfile.writerow(["datetime_stamp","filename","raw_overlap","lhood_overlap", \
               "loglhood_overlap", "total_marks","after_marks", "total_rows"])

    countfile.writerow(["# AUTO"])
    countfile.writerow(header_info)



if __name__ == "__main__":
    # assume 1 argument : the filename of the input datafile
    # argument 2 : the starting row of anomaly detection counter
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', '--name', help="name of datafile", default='aasane01_03.csv')
    parser.add_argument('-s', '--startrow', type=int, help="starting row for anomaly counts", default=0)
    parser.add_argument('-g', '--aggregate', type=int, help="add aggregation step", default=1)
    parser.add_argument('-t', '--aggfunc', help="set aggregation function (mean,sum)", default="sum")
    args = parser.parse_args()


    # alpha = [0.005, 0.05, 0.5]
    # maxboost = [1.0, 2.0]
    # values = [100, 200, 400, 800]
    # time = [None, {'fieldname': 'timestamp', 'name': 'timestamp', 'type': 'DateEncoder', \
    #                                                                 'weekend': ( 21, 1)}]

    # alpha = [0.0095]
    # maxboost = [1.0, 2.0]
    # values = [250, 400]
    # time = [None]#, {'fieldname': 'timestamp', 'name': 'timestamp', 'type': 'DateEncoder', \
    # pamlen = [1, 2, 3]

    alpha = [0.0095, 0.0993]
    maxboost = [1.0]
    values = [250, 600]
    time = [None, {'fieldname': 'timestamp', 'name': 'timestamp', 'type': 'DateEncoder', \
                                                                     'timeOfDay': ( 21, 3)}]
    pamlen = [1, 3]

    starttime = datetime.now()

    for af, mb, val, tm, pam in itertools.product(alpha, maxboost, values, time, pamlen):

        # new batch entry in log
        header = ["# alpha: %.3f \tmBoost: %.1f \tN: %d \tWEnd: %s \tpam: %d" % (af, mb,
                                                val, (tm!=None), pam)]
        makeLogHeader(header)

        for case in mig.CASE_NAMES[:-2]:
            model = setActiveModelParams(af, mb, val, tm, pam)
            current_stats = run_experiment(model, case, args.startrow, args.aggregate, args.aggfunc)

            makeLogEntry(current_stats)
            print header
            print "File: %s \tdone at: %s" % (case, datetime.now())




    print ("Time elapsed: ", (datetime.now()-starttime).total_seconds()/60.0)
