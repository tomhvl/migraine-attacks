MODEL_PARAMS = \
{ 'aggregationInfo': { 'days': 0,
                       'fields': [],
                       'hours': 0,
                       'microseconds': 0,
                       'milliseconds': 0,
                       'minutes': 0,
                       'months': 0,
                       'seconds': 0,
                       'weeks': 0,
                       'years': 0},
  'model': 'CLA',
  'modelParams': { 'anomalyParams': { u'anomalyCacheRecords': None,
                                      u'autoDetectThreshold': None,
                                      u'autoDetectWaitRecords': None},
                   'clParams': { 'alpha': 0.00844763845387081,
                                 'regionName': 'SDRClassifierRegion',
                                 'steps': '1',
                                 'verbosity': 0},
                   'inferenceType': 'TemporalAnomaly',
                   'sensorParams': { 'encoders': { '_classifierInput': { 'classifierOnly': True,
                                                                         'clipInput': True,
                                                                         'fieldname': 'value',
                                                                         'maxval': 4500,
                                                                         'minval': 0,
                                                                         'n': 521,
                                                                         'name': '_classifierInput',
                                                                         'type': 'ScalarEncoder',
                                                                         'w': 21},
                                                   u'value': { 'clipInput': True,
                                                               'fieldname': 'value',
                                                               'maxval': 4500,
                                                               'minval': 0,
                                                               'n': 373,
                                                               'name': 'value',
                                                               'type': 'ScalarEncoder',
                                                               'w': 21},
                                                   u'timestamp_dayOfWeek': None,
                                                   u'timestamp_timeOfDay': { 
                                                   				'fieldname': 'timestamp',
                                                            	'name': 'timestamp',
                                                            	'timeOfDay': ( 21, 3.090344152692538),
                                                                'type': 'DateEncoder'},
                                                   u'timestamp_weekend': {
                                                   				'fieldname': 'timestamp',
                                                                'name': 'timestamp',
                                                                'type': 'DateEncoder',
                                                                'weekend': ( 21, 1)}},
                                     'sensorAutoReset': None,
                                     'verbosity': 0},
                   'spEnable': True,
                   'spParams': { 'columnCount': 2048,
                                 'globalInhibition': 1,
                                 'inputWidth': 0,
                                 'maxBoost': 1.0,
                                 'numActiveColumnsPerInhArea': 40,
                                 'potentialPct': 0.8,
                                 'seed': 1956,
                                 'spVerbosity': 0,
                                 'spatialImp': 'cpp',
                                 'synPermActiveInc': 0.05,
                                 'synPermConnected': 0.1,
                                 'synPermInactiveDec': 0.0003},
                   'tpEnable': True,
                   'tpParams': { 'activationThreshold': 14,
                                 'cellsPerColumn': 32,
                                 'columnCount': 2048,
                                 'globalDecay': 0.0,
                                 'initialPerm': 0.21,
                                 'inputWidth': 2048,
                                 'maxAge': 0,
                                 'maxSegmentsPerCell': 128,
                                 'maxSynapsesPerSegment': 32,
                                 'minThreshold': 12,
                                 'newSynapseCount': 20,
                                 'outputType': 'normal',
                                 'pamLength': 2,
                                 'permanenceDec': 0.1,
                                 'permanenceInc': 0.1,
                                 'seed': 1960,
                                 'temporalImp': 'cpp',
                                 'verbosity': 0},
                   'trainSPNetOnlyIfRequested': False},
  'predictAheadTime': None,
  'version': 1}
