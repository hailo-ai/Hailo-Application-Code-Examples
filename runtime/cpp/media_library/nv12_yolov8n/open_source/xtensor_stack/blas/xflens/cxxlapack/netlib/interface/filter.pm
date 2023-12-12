#! /usr/bin/perl -w

##
##  for i in ../lapack/*.f; do cat $i | ./filter.pm impl; done > dummy.in.cc
##
##  for i in ../lapack/*.f; do cat $i | ./filter.pm header; done > lapack.in.h
##

use strict;

die unless $#ARGV==0;


my $passThrough = -1;

$passThrough = 0 if $ARGV[0] eq "header";
$passThrough = 1 if $ARGV[0] eq "impl";

die if $passThrough == -1;

my @lines;

my $first = 1;
while (my $line=<STDIN>) {
    print STDERR $line if $first;
    $first = 0;
    chomp($line);
    if ($line =~ /================/) {
        last;
    }
    if ($line =~ /^[^\$]*\$\s*(.*)/) {
        $lines[$#lines] .= $1;
        next;
    }
    push(@lines, $line);
}

my %args;
my $subroutine;
my $returnType = "void";

foreach my $line (@lines) {
    if ($line =~ /^\s*\*/) {
        next;
    }
    if ($line =~ /CHARACTER[*1]*\s(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "char";
        }
    } elsif ($line =~ /INTEGER(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(",", $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "INTEGER";
        }
    } elsif ($line =~ /LOGICAL(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "LOGICAL";
        }
    } elsif ($line =~ /DOUBLE PRECISION(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "DOUBLE";
        }
    } elsif ($line =~ /^\s*COMPLEX[*]16\s(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "DOUBLE_COMPLEX";
        }
    } elsif ($line =~ /^\s*COMPLEX(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "FLOAT_COMPLEX";
        }
    } elsif ($line =~ /^\s*REAL(.*)/) {
        my $vars = $1;
        $vars =~ s/\([^)]*\)//g;
        my @vars = split(/,/, $vars);
        for my $var (@vars) {
            $var =~ s/\s//g;
            $args{$var} = "FLOAT";
        }
    }

    if ($line =~ /SUBROUTINE (.*)/) {
        $subroutine = $1;
        $subroutine =~ s/ //g;
    }
    if ($line =~ /(.*) FUNCTION (.*)/) {
        $returnType = $1;
        $subroutine = $2;
        $subroutine =~ s/ //g;

        $returnType =~ s/^\s*//;
        $returnType =~ s/\s*$//;
        if ($returnType =~ /DOUBLE PRECISION/) {
            $returnType = "DOUBLE";
        } elsif ($returnType =~ /REAL/) {
            $returnType = "FLOAT";
        } elsif ($returnType =~ /INTEGER/) {
            $returnType = "INTEGER";
        } elsif ($returnType =~ /LOGICAL/) {
            $returnType = "LOGICAL";
        } else {
            $returnType = "UNKNOWN";
        }
    }
}

my %args_non_const;
foreach my $line (@lines) {
    for my $var (keys %args) {
        if ($line =~ /\s*\*\s*$var\s*\(.*output.*\)/) {
            $args_non_const{$var} = $var;
            next;
        }
        if ($line =~ /\s*\*\s*$var\s*\(.*workspace.*\)/) {
            $args_non_const{$var} = $var;
        }
    }
}

for my $var (keys %args) {
    unless ($args_non_const{$var}) {
        $args{$var} = "const " . $args{$var};
    }
}

if ($subroutine) {
    my $name;
    my @vars;

    $subroutine =~ /([^(]*)\((.*)\)/;
    $name = $1;
    @vars = split(",", $2);

    for my $var (@vars) {
        unless ($args{$var}) {
            print "// warning: $var has unknown type\n";
            $args{$var} = "UNKNOWN";
        }
    }

    my $declName = "LAPACK_IMPL(" . lc($name) . ")(";

    my $paddingWidth = 0;
    for my $var (@vars) {
        if (length($args{$var})>$paddingWidth) {
            $paddingWidth = length($args{$var});
        }
    }
    $paddingWidth = (int(($paddingWidth + length($declName)) / 4) + 1)*4
                  - int(length($declName)/4)*4;

    my @param;
    for my $var (@vars) {
        my $padding = " " x ($paddingWidth - length($args{$var}));
        push(@param, "$args{$var}$padding *$var");
    }

    my $padding = " " x length($declName);

    print "//-- " . lc($name) . " " . "-" x (80 - 6 -length($name)) . "\n";
    print "$returnType\n";
    print $declName . join(",\n$padding", @param) . ")";

    if ($passThrough) {
        print "\n";
        print "{\n";

        print " " x 4 . "DEBUG_LAPACK_STUB(\"" . lc($name) . "\");\n";

        if (lc($name) =~ /extended/) {
            print " " x 4 . "ASSERT(0);\n";
            print " " x 4 . "/*\n";
        }

        my $defName = "LAPACK_IMPL(" . lc($name) . ")(";

        if ($returnType ne "void") {
            $defName = "return " . $defName;
        }

        $defName = " " x 4 . $defName;
        $padding = " " x length($defName);
        print $defName . join(",\n$padding", @vars) . ");\n";

        if (lc($name) =~ /extended/) {
            print " " x 4 . "*/\n"
        }

        print "}\n\n";
    } else {
        print ";\n\n";
    }
}