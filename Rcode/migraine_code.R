

fn_lavg <- function(x, winsize = length(x)) {
  res = c()
  avx = sum(x[1:winsize - 1])
  winstart = 0
  winstop = winsize
  for (elem in x[winsize:length(x)]) {
    avx = avx + elem
    if (winstop - winstart > winsize) {
      winstart = winstart + 1
      avx = avx - x[winstart]
    }
    res = c(res, avx / (winstop - winstart))
    winstop = winstop + 1
  }
  res
}

fn_wskew <- function(x, winsize = length(x)) {
  res = c()
  winstart = 0
  winstop = winsize
  for (elem in x) {
    if (winstop - winstart > winsize) {
      winstart = winstart + 1
    }
    res = c(res, skewness(x[winstart:winstop]))
    winstop = winstop + 1
  }
  res
}

# find continuous ( < k zero) minsize>sequence<maxsize, where a sequence count starts at 1st non-zero entry
# returns list of pairs begin:end
# x = data sequence
# k = min number of consecutive zeros to decide cut-off/boundary
# minsize = minimum required continous values
# maxsize = max allowed chunk allowed
fn_getSequence <-
  function(x,
           k = 2,
           minsize = 50,
           maxsize = length(x)) {
    
    winstart = 1
    winstop = 1
    kcount = 0
    
    skip = FALSE
    viable = FALSE
    wbegin = c()
    wend = c()
    
    while (winstop < length(x)) {
      if (x[winstop] == 0) {
        kcount = kcount + 1
        
        if (kcount == k) {
          skip = TRUE
          if (viable) {
            # remember this window less the zeros
            wbegin = c(wbegin, winstart)
            wend = c(wend, winstop-k)
          }
        }
      } else{
        kcount = 0
        if (skip) {
          winstart = winstop
          skip = FALSE
        }
      }
      
      winstop = winstop + 1
      
      # do we have a minimum sized window of leagl values?
      viable = (winstop - kcount - winstart >= minsize) && !skip
      
      if (winstop - kcount - winstart >= maxsize) {
        #skip = TRUE
        if (viable) {
          wbegin = c(wbegin, winstart)
          wend = c(wend, winstop-(kcount%%k))
          viable = FALSE
          winstart = winstop+1
        }
      }
      

    }
    
    if (viable) {
      wbegin = c(wbegin, winstart)
      wend = c(wend, winstop)
    }
    
    list("begin"=wbegin, "end"=wend)
    
  }


# find skewness between begin:end pairs of points from a list(beginVec, endVec)
# should be used with fn_getSequence() and uses $begin $end element ids
fn_skewness <- function(x, xs) {
  res = c()
  for (i in 1:length(xs$begin)) {
    res = c(res, skewness(x[xs$begin[i]:xs$end[i]]))
  }
  
  list("begin"=xs$begin, "end"=xs$end, "skw"=res)
  res
}

# find kurtosis between begin:end pairs of points from a list(beginVec, endVec)
# should be used with fn_getSequence() and uses $begin $end element ids
fn_kurtosis <- function(x, xs) {
  res = c()
  for (i in 1:length(xs$begin)) {
    res = c(res, kurtosis(x[xs$begin[i]:xs$end[i]]))
  }
  
  list("begin"=xs$begin, "end"=xs$end, "kur"=res)
  res
}

fn_execute <- function(x, xs, FUN) {
  res = c()
  for (i in 1:length(xs$begin)) {
    res = c(res, FUN(x[xs$begin[i]:xs$end[i]]))
  }
  
  list("begin"=xs$begin, "end"=xs$end, "kur"=res)
  res
}


# find largest chunk before and after list of points, within winradius
# takes a list of 2 paired vectors (from fn_getSequence)
fn_getMaxChunk <- function(x, begin, end, minsize = 50, maxsize = end-begin) {
  # find longest recorded period
  best = c(0,0)
  foer = fn_getSequence(x[begin:end], k = 2, minsize, maxsize)
  if (length(foer$end) > 0) {
    for (i in 1:length(foer$begin)) {
      if(foer$end[i]-foer$begin[i] > best[2]-best[1]) {
        best[1] = foer$begin[i]+begin-1
        best[2] = foer$end[i]+begin-1
      }
    }
  }
  
  # return a window begin,end  of biggest size in given partition
  best
}

# data, marks, > k zeros allowed, period = max windowsize,
# return vector period(min winsize) before mark, after mark
fn_batchSk <- function(data,
                       marks,
                       minsize = 50,
                       maxsize = 60,
                       period = 120,
                       margin = 0) {
  
  begin = c()
  end = c()
  
  for (elem in marks) {
    tmp = fn_getMaxChunk(data, elem-period, elem+margin, minsize, maxsize)
    begin = c(begin, tmp[1])
    end = c(end, tmp[2])
    
    tmp = fn_getMaxChunk(data, elem-margin, period+elem, minsize, maxsize)
    begin = c(begin, tmp[1])
    end = c(end, tmp[2])
  }
  
  list("begin"=begin, "end"=end)
}


# divide up into chunks, 1 for each 24hrs (1440mins)
# returns a matrix with |data|/size cols
fn_splitByDays <- function(data, start=1, size=1440) {
  
  A = matrix(data[start:size], nrow = size)
  i = start + size = 
  while (i+size < length(data)) {
      A = cbind(A, c(data[i:(i+size+1)]))
      i = i + size
  }
  
  A
  
}



# extract a sequence of datapoints around labels given total size and position of mark in sequence
# example: size=60, pos=1, will return a window starting at marked point + 60 points
#          size=61, pos=30, will return a window with marked being the 30th index in that window (middle in this case)
# expects a dataframe with "marked" attribute column "M"= marked
fn_extractByMarked <- function(data, size=60, mpos=1) {
  
  idx = which(data$marked =='M')
  
  # build up a matric with chosen sequences as columns
  resMatrix = matrix(nrow = size, ncol=0)
  for (i in idx) {
    resMatrix = cbind(resMatrix, data$value[(i-mpos+1):(i-mpos+1+size)])
  }
  
  resMatrix
  
}




# return a subset of rows/examples/instances from data, starting at datetime tstart (POSIX)
# for a period of tmin minutes. NOTE: expects data to have a POSIX timestamp attribute/column named "timestamp"
# ex: tmp = fn_extractPeriod(df, as.POSIXlt('2005-08-19 04:00:00'), tmin=24*60)
fn_extractPeriod <- function(data, tstart = data$timestamp[1], tmin = 60) {
  subset(data, data$timestamp >= as.POSIXct(tstart) & data$timestamp < as.POSIXct(tstart+tmin*60))
}



# useful libs
library(dplyr)
# for ts, with rollapply, https://cran.r-project.org/web/packages/zoo/vignettes/zoo-quickref.pdf
library(zoo)
# maybe caret package ?


# ************* steps to import and further prep stp_ data **********
# 
# # read in data from stp-csv file
# filename = "stm_aasane18_modified.csv"
# df = read.csv(filename, header = F, col.names = c("timestamp", "value", "marked"), skip = 3)
# 
# # transform string timesatmps into POSIX 
# df$timestamp <- strptime(x = as.character(df$timestamp), format="%Y-%m-%d %H:%M")
# # extract weekdays information 
# df$day = (weekdays(as.Date(df$timestamp), abbreviate = T))
# 
# # examples of transforming/managing datetime
# tmp.start = strptime(as.character('2005-08-15 04:00:00'), format="%Y-%m-%d %H:%M")
# tmp = subset(df, df$timestamp > as.POSIXct(tmp.start) & df$timestamp < as.POSIXct(tmp.stop))
# tmp.delta = difftime(tmp.start, df$timestamp[1], units = "mins")
# which(tmp$marked !='M')
# 
# #extract hour from POSIXlt variable
# df$timestamp[1]$hour
# format(df$timestamp[1], '%H')
# 
# # (tmp is a data frame with POSIXlt timestamp, value, marks, ..)
# # select the datapoints with MINS field = "09"
# tmp$timestamp[which(format(tmp$timestamp, '%M') == "09")]
# 



# TODO:   to show how the daily data differs from day to day
#         -> also do this for same day of each week
#         -> same day of different subjects (no anomalies)
#         -> 2-4h periods for anomalies (of the same subject)
#         -> + other such combinations
#         -> possibly check labeled sequences and supply flag to show/skip those only?

# take snippets of daily data -> table/matrix
# calculate rolling mean/median for each of equal order
# plot the results aligned on top of each other
# and calculate correlation (or similar statistic that compares them)

# extract daily sequences of data from a single subject data frame, by timestamp hour/min and duration
# expects data frame (data) to have columns timestamp (at least HH:MM) & value
# returns a matrix where each column is a described sequence
op_dailySingleSubject <- function(data, startHour="", startMin="", durationMins = 24*60) {
  
  matchStr =""
  startTime = ""
  
  # build a search string representation for the timestamped time
  if (startHour != "") {
    matchStr = "%H"
    startTime = startHour
    
    if (startMin != "") {
      matchStr = "%H:%M"
      startTime = stringi::stri_join(startHour, ":", startMin)
    }
    
  }else if (startMin != "") {
    matchStr = "%M"
    startTime = startMin
  }
    
  
  # build up a matric with chosen sequences as columns
  resMatrix = matrix(nrow = durationMins, ncol=0)
  for (i in 1:length(data$timestamp)) {
    
    if (format(data$timestamp[i], matchStr) == startTime ) {
      print(data$timestamp[i])
      print(i)
      
      resMatrix = cbind(resMatrix, data$value[i:(i+durationMins)])
      i = i + durationMins
    }
  }
  
  resMatrix
}




# **************** plots based on extracted vectors/matrices ************************************

# plot consecutive days overlapping rolling means (moving averages), distinct colors for weekends
# accepts a matrix with columns for each 24hrs period (as output by fn_dailySingleSubject)
# startDay 1=monday... 7=sunday,  mask=days to show in a week C(1,1,1,1,1,0,0) shows working days only
pl_mavg <- function(vdata, begin=1, end=1, winsize=180, mask=c(1,1,1,1,1,1,1), startDay=1) {
  
  scaleFactor = 1/60
  high = max(vdata[,begin:end])
  wide = dim(vdata)[1]
  
  clr = c(1,1,1,1,1,6,6)
  plot.ts(0, xlim=c(0,wide-winsize), ylim=c(0, high*scaleFactor/4))
  
  for (i in begin:end) {
    if (mask[startDay]) {
      lines(smooth(rollmeanr(vdata[,i]*scaleFactor, winsize)), lwd=1, col=clr[startDay])
    }
    
    startDay = (startDay%%7)+1 
  }
}

# same as above but with median
pl_mmed <- function(vdata, begin=1, end=1, winsize=180, mask=c(1,1,1,1,1,1,1), startDay=1) {
  
  scaleFactor = 1/60
  high = max(vdata[,begin:end])
  wide = dim(vdata)[1]
  
  clr = c(1,1,1,1,1,6,6)
  plot.ts(0, xlim=c(0,wide-winsize), ylim=c(0, high*scaleFactor/4))
  
  for (i in begin:end) {
    if (mask[startDay]) {
      lines(smooth(rollmedian(vdata[,i]*scaleFactor, winsize)), lwd=2, col=clr[startDay])
    }
    
    startDay = (startDay%%7)+1 
  }
}


# function to generate a plot of (function) differences between 2 consecutive windows
# sliding window divided in 2 equal parts, calculate skew/kurtosis on each and find difference
# winsize = size of EACH of the 2 windows
fn_dualComp <- function(vdata, winsize=60, FUN) {
  
  rwin = rollapply(vdata[winsize:length(vdata)], winsize, by=1, FUN)
  lwin = rollapply(vdata[1:(length(vdata)-winsize)], winsize, by=1, FUN)
  
  res = c()
  for (i in 1:length(lwin)) {
    res = c(res, lwin[i] - rwin[i])
  }
  
  res
}



# plot a simple overlapping graph
#cls = colors(TRUE)
#plot.ts(rollmean(resMatrix[,1], 181), xlim=c(durationMins,0), ylim=c(0,1000), type="l")
#plot(rollapply(resMatrix[,1], 225, kurtosis),xlim=c(durationMins,0), ylim=c(0,200), type="l")
#for (i in 2:dim(resMatrix)[2]) {
#  lines(rollapply(resMatrix[,i], 225, kurtosis), type="l", col=cls[140+i*5])
#}


# divide file into separate chunks which will be the samples
# takes a vector, a chunk size
# returns a matrix of r x c (r=number of chunks) (cols=number of points == chunksize)
fn_createChunks <- function(data, chunksize=60) {
  
  resMatrix = matrix(nrow = 0, ncol=chunksize)
  for (i in seq(1,length(data), chunksize)) {
    print(i)
    
    resMatrix = rbind(resMatrix, data[i:(i+chunksize)])

  }
  
  resMatrix
  
}




# example

n <- 50
idx <- sample(1:432, n)
sample2 <- chunks[idx,]
hc = hclust(dist(sample2), method = "ave")
observed = idx
plot(hc, labels=observed)






