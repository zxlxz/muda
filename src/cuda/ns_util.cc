#include "ns_util.h"

auto File::readToString(NS::String* path) -> NS::String* {
  auto file = ::fopen(path->utf8String(), "r");
  if (!file) {
    return nullptr;
  }

  ::fseek(file, 0, SEEK_END);
  const auto size = ::ftell(file);
  ::fseek(file, 0, SEEK_SET);
  if (size <= 0) {
    ::fclose(file);
    return nullptr;
  }

  char* buf = static_cast<char*>(::malloc(size + 1));
  ::memset(buf, 0, size + 1);
  ::fread(buf, 1, size, file);
  ::fclose(file);

  auto res = NS::String::string(buf, NS::UTF8StringEncoding);
  ::free(buf);
  return res;
}
