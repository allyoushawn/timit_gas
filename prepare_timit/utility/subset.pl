#!/usr/bin/perl

my $n = $ARGV[0];
my @set;
for(my $i = 0; $i < $n; $i++) {
    push @set, 0;
}
my $k = $ARGV[1];
my @s = split(/:/, $k);
for(my $i = 0; $i <= $#s; $i++) {
    $set[$s[$i]] = 1;
}

my $idx = 1;
while(<STDIN>) {
    chomp;
    if($set[$idx] != 0) { print "$_\n";}
    if($idx == $n) { $idx = 1; }
    else { $idx++; }
}
