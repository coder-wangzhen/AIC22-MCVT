#ifndef _COMMON_DEFINE_
#define _COMMON_DEFINE_

#include<string>

typedef struct res_info
{
  std::string message;
  bool ret;
  res_info()
  {
    message = "";
    ret = true;
  }
} RET_INFO;

#endif