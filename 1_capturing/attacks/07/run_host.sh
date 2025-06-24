redis-cli -h $MASTER_IP -p 30007 eval "return bit.tohex(65535, -2147483648)" 0
