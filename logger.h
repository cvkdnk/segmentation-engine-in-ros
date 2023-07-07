//
// Created by 17904 on 2022/12/10.
//

#ifndef ROS_ENGINE_LOGGER_H
#define ROS_ENGINE_LOGGER_H
#include "logging.h"
namespace cpgnet {
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
    extern LogStreamConsumer gLogFatal;
    void setReportableSeverity(Logger::Severity severity);
} // namespace cpgnet

#endif //ROS_ENGINE_LOGGER_H
