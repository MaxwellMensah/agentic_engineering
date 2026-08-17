# Linux Stack Cheatsheet

### Layer 1: Kernel, Namespaces & cgroups
* **Command:** `sudo lsns -t net`
* **Purpose:** Displays all active isolated network namespaces on the kernel.
* **Key Finding:** Proves Docker containers are isolated processes (sharing kernel namespaces) rather than full Virtual Machines.

### Layer 2: Networking
* **Command:** `sudo iptables -t nat -L DOCKER -n -v`
* **Purpose:** Inspects Docker's Network Address Translation (NAT) rules.
* **Key Finding:** Shows how Docker uses iptables to route host ports (e.g. 8080) directly to container internal IP addresses.

### Layer 3: Storage
* **Command:** `lsblk -f`
* **Purpose:** Lists block storage devices and underlying formatted filesystems (`ext4`, `xfs`).
* **Key Finding:** Maps physical storage partitions to mountpoints (`/`) where container image overlay layers reside.

### Layer 4: Application & System Calls
* **Command:** `strace -e trace=openat,write python3 -c "print('Hello Linux')"`
* **Purpose:** Intercepts low-level system calls made by a process to the Linux kernel.
* **Key Finding:** Reveals how userland processes request file access (`openat`) and I/O execution (`write`) from the kernel.