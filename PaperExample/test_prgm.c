#include <stdio.h>

int main(void) {
    int x;
    scanf("%d", &x);
    int y = x * 2;
    if (y > 15) {
        printf("Condition Satisfied \n");
    } else {
        printf("Condition Not Satisfied \n");
    }
    printf("Program End \n");
}