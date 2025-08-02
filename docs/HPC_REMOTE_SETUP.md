# Accessing CWRU Pioneer HPC System Remotely via SSH (PuTTY)

## Step 1: Set up DUO Authentication for VPN Access

### 1. Enroll in DUO (if not already done):

> - Go to [case.edu/utech/duo](https://case.edu/utech/duo) and follow instructions to register your device (phone/tablet/hardward token)
> - This is required for FortiClient VPN authentication.

---

## Step 2: Install and Configure FortiClient VPN

### 1. Download FortiClient VPN:

- Visit [case.edu/utech/help/forticlient-vpn](https://case.edu/utech/help/forticlient-vpn)
- Download the **FortiClient VPN** software for your specific device.

### 2. Install & Configure VPN

- Run the installer and complete setup
- Open FortiClient and configure new connection:
  - **Connection Name**: `CWRU VPN` (or any name)
  - **Remote Gateway**: `vpn.case.edu`
  - **Customize Port**: `443`
  - Enable "**Save Credentials**" (optional)
- Click **Save**

### 3. Connect to VPN:

- Enter your **CWRU Network ID** (e.g., `jxh369`) and password.
- Complete **DUO two-factor authentication** when prompted (approve via phone/device)
- Once connected, you'll see a confirmation message.

---

## Step 3: Install PuTTY (SSH Client)

### 1. Download PuTTY:

- If not installed, download from [https://www.putty.org](https://www.putty.org)
- Run the installer (or use the portable version).

## 2. Open PuTTY:

- Launch PuTTY from the Start Menu

---

## Step 4: Configure PuTTY for Pioneer HPC

### 1. Enter Connection Details:

- **Host Name (or IP address)**: `pioneer.case.edu`
- **Port**: `22`
- **Connection Type**: SSH

### 2. Optional: Save Session (for future use):

- Under "**Saved Sessions**", type `Pioneer HPC` and click **Save**

### 3. Click "Open" to initiate the connection

---

## Step 5: Log In via SSH

### 1. Enter Credentials:

- When prompted, enter your **CWRU Network ID** (e.g., `jxh369`)
- Enter your password (same as VPN/CWRU login)
- Complete DUO authentication again if required

### 2. Successful Login:

- You should now see the **Pioneer HPC command-line interface**

---

## Step 6: Disconnecting

### 1. Exit SSH Session:

- Type `exit` or `logout` in the terminal

### 2. Disconnect VPN:

- Close PuTTY and disconnect FortiClient VPN when done.

---

## Troubleshooting Tips

### VPN Fails?

- Ensure DUO is set up correctly
- Try reconnecting or restarting FortiClient VPN

### PuTTY Connection Refused?

- Verify VPN is active (`vpn.case.edu` shows "**Connected**")
- Check `pioneer.case.edu` and port `22` are correct

## DUO Not Prompting?

- Ensure your device is registered in DUO


## Extra Help on CWRU HPC Systems

[https://sites.google.com/a/case.edu/hpcc/](https://sites.google.com/a/case.edu/hpcc/)