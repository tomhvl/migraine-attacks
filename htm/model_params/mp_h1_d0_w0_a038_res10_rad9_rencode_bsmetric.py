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
                                      u'autoDetectWaitRecords': 5030 },
                   'clParams': { 'alpha': 0.035828933612157998,
                                 'verbosity': 0,
                                 'regionName': 'SDRClassifierRegion',
                                 'steps': '1'},
                   'inferenceType': 'TemporalAnomaly',
                   'sensorParams': { 
                   				 'encoders': { u'value': { 'clipInput': True,
                                                               'fieldname': 'value',
                                                               'maxval': 4000,
                                                               'minval': 0,
                                                               'resolution': 10.0,
                                                               'name': 'value',
                                                               'type': 'RandomDistributedScalarEncoder',
                                                                },
                                                   u'timestamp_dayOfWeek': None,
                                                   u'timestamp_timeOfDay': { 
                                                   				'fieldname': 'timestamp',
                                                            	'name': 'timestamp',
                                                            	'timeOfDay': ( 21, 9.49),
                                                                'type': 'DateEncoder'},
                                                   u'timestamp_weekend': None
                                                                },
                                     'sensorAutoReset': None,
                                     'verbosity': 0},
                   'spEnable': True,
                   'spParams': { 'columnCount': 2048,
                                 'globalInhibition': 1,
                                 'inputWidth': 0,
                                 'boostStrength': 0.0,
                                 'numActiveColumnsPerInhArea': 40,
                                 'potentialPct': 0.8,
                                 'seed': 1956,
                                 'spVerbosity': 0,
                                 'spatialImp': 'cpp',
                                 'synPermActiveInc': 0.003,
                                 'synPermConnected': 0.2,
                                 'synPermInactiveDec': 0.0005},
                   'tpEnable': True,
                   'tpParams': { 'activationThreshold': 13,
                                 'cellsPerColumn': 32,
                                 'columnCount': 2048,
                                 'globalDecay': 0.0,
                                 'initialPerm': 0.21,
                                 'inputWidth': 2048,
                                 'maxAge': 0,
                                 'maxSegmentsPerCell': 128,
                                 'maxSynapsesPerSegment': 32,
                                 'minThreshold': 10,
                                 'newSynapseCount': 20,
                                 'outputType': 'normal',
                                 'pamLength': 3,
                                 'permanenceDec': 0.1,
                                 'permanenceInc': 0.1,
                                 'seed': 1960,
                                 'temporalImp': 'cpp',
                                 'verbosity': 0},
                   'trainSPNetOnlyIfRequested': False},
  'predictAheadTime': None,
  'version': 1}
