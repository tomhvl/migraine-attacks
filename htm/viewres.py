# evaluate the output from activity_experiment output statistics

import sys, os.path
import argparse

import pandas as pd
from datetime import datetime


# overlap period size (as described in activity experiment
OVERLAP_PERIOD = 120

START_ROW = 4320    # original singlestep 4320, agg new 1000

def summarize(filename, step, stype):

    anomData = open(filename,"rb")
    anomData.readline()

    bestPrec = bestRecall = bestAcc = 0.01
    bestPrecLine = bestRecLine = bestAccLine = ""
    precision = recall = accuracy = 0.0

    skipped = 0
    stepFlag = False
    counter =0
    beforeAfterRatio = 0.0

    results = []

    for row in anomData:
        # col index 2,3,4 have the raw anomaly scores (pre/post/total)
        # 5,6,7 (likelihood), 8,9,10 (log_likelihood)
        # 11 marks_total, 12 marks_after, 13 total_rows
        # totals adjusted for non-zero division
        if row[0] == "#":
            stepFlag = "step=" in row
            continue

        # differentiate between stepped and non stepped (aggregated runs)
        if stepFlag != step:
            continue

        # count used instances
        counter += 1

        row = row.translate(None, '["]')
        row = row.split(',')

        rowsTotal = float(row[13]) - START_ROW

        realPositives = float(row[12])
        #realNegatives = rowsTotal - realPositives

        pTotal = float(row[4+stype])         # this is the same as (tp + fp)
        tp = float(row[3+stype])
        fp = pTotal - tp
        tn = abs(rowsTotal - pTotal)

        beforeTP = int(row[2+stype])


        # a sequence/window is one instance/example for simplification
        totalExamples = int(rowsTotal-START_ROW)/OVERLAP_PERIOD
        #print rowsTotal, totalExamples

        # NOTE: it is possible to get multiple hits in one labeled area!
        try:
            accuracy = round((tp+tn) / rowsTotal, 4)
            #lhood_accuracy = (rowsTotal - row[7] + row[6]) / rowsTotal
            #log_accuracy = (rowsTotal-row[10] + row[9]) / rowsTotal

            # tp/allPositives ,if (tp > realPositives)
            # this is not correct unless we only count 1 hit per labelled area (ATM: counting multiple)
            recall = round(tp / max(realPositives, tp), 3)

            # new fixed recall calculation, has appended [recall scores]
            
            if len(row) > 14+stype/3:
                recall = round(float(row[14+stype/3]) / realPositives, 3)

            # precision, tp/tp+fp
            precision = round(tp / pTotal, 3)

            # tn true neg when no anomaly exists and none is found
            # fn false neg when anomaly is not detected
            # tp anomaly detected correctly
            # fp non-anomaly detected as anomaly

            if bestPrec < precision and fp > realPositives and realPositives > 0:
                bestPrec = precision
                bestPrecLine = [row[0], row[1][:-13], accuracy, precision, recall, round(realPositives/rowsTotal, 5), len(row)>14]
                #results.append(bestPrecLine)
    
            if bestRecall < recall and fp > realPositives and realPositives > 0:
                bestRecall = recall
                bestRecLine = [row[0], row[1][:-13], accuracy, precision, recall, round(realPositives/rowsTotal, 5), len(row)>14]
                #results.append(bestRecLine)
    
            if bestAcc < accuracy and fp > realPositives and realPositives > 0:
                bestAcc = accuracy
                bestAccLine = [row[0], row[1][:-13], accuracy, precision, recall, round(realPositives/rowsTotal, 5), len(row)>14]
                #results.append(bestAccLine)
    
            results.append([row[0], row[1][:-13], accuracy, precision, recall, round(realPositives/rowsTotal, 5), len(row)>14])
                        
            beforeAfterRatio = beforeAfterRatio + ((beforeTP+1)/(tp+1))
            
            
        except ZeroDivisionError as detail:
            # print 'Skipping : ', row[0],row[1]
            skipped += 1
        
    # output best result
    scoreType = ["raw", "likelihood", "logLikelihood"]

    print "[Run, ID, Accuracy, Precision, Recall, Prevalence] :    Aggregation: %s" % (step)

    print "Best "+scoreType[stype/3]+" anomaly accuracy score: ", bestAcc
    print "  in line: ", bestAccLine 
    print "Best "+scoreType[stype/3]+" anomaly precision score: ", bestPrec
    print "  in line: ", bestPrecLine
    print "Best "+scoreType[stype/3]+" anomaly recall score: ", bestRecall
    print "  in line: ", bestRecLine

    print "Average before:after ratio:", round(beforeAfterRatio/(counter-skipped),2)
    print "Number of ignored results: ", skipped

    res = pd.DataFrame(results, columns=['stamp', 'name', 'acc', 'prec',
                                         'recall', 'preval', 'newRecall'])
    res.to_csv("summary_type%d_agg%d.csv" % (stype/3+1, step), sep=',')
    
    anomData.close()

    print "Done: summary_scoretype=%d aggregation=%r" % (stype/3+1, bool(step))


if __name__ == "__main__":
	# assume 1 argument : the filename of the input datafile
    # argument 2 : the starting row of anomaly detection counter
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--name', help="name of datafile", default='anomaly_count_stats_new.csv')
    parser.add_argument('-s', '--step', type=bool, help="aggregation flag", default=False)
    parser.add_argument('-t', '--stype', type=int, help="score type, 1=raw, 2=likelihood, 3=logLhood", default=2)

    args = parser.parse_args()
    
    summarize(args.name, args.step, (args.stype-1)*3)
