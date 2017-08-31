#!/usr/bin/perl

while(<STDIN>) {
    my @arr = split(/\s+/, $_);
    printf("%-20s", $arr[0]);
    for(my $i = 1; $i <= $#arr; $i++) {
        if(($arr[$i] =~ /^\[.*\]$/)&&(length($arr[$i]) > 6)) {
            for(my $j = 0; $j < length($arr[$i]); $j += 6) {
                printf(" %s", substr($arr[$i], $j, 6));
            }
        }
        else {
            printf(" %s", $arr[$i]);
        }
    }
    printf("\n");
}

