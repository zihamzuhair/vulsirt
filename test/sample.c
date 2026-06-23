#include <string.h>

void unsafe_copy(char *dst, const char *src) {
    strcpy(dst, src);
}
