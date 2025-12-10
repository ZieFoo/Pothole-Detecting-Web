To ensure that python3.py works, make sure the folder contains appv2.py, best.onnx, and the templates folder which consists of indexv2.html, viewerv2.html and pothole_mapv2.html.

To ensure that all required Python libraries are installed, create a virtual environment using Python version 3.10.19. After activating it, run the following command:

```bash
pip install -r requirements.txt
```

To run the program, open a terminal, navigate to the folder where appv2.py is located, and run:

```bash
python appv2.py
```

To run the server using srv.us, open another terminal and use:

```bash
ssh srv.us -R 1:127.0.0.1:5000
```

You will receive an HTTPS link. Copy that link and open it on your device. The program should now run successfully.

First-Time Use: Email Authentication

The first time you use srv.us, it may ask you to enter your email address.
This is normal — srv.us uses your email for lightweight verification instead of creating accounts or passwords.

Enter a valid email (a temporary email is fine if you prefer).

srv.us uses this email to generate a unique, temporary public hostname for your session.

You normally don’t need to click any confirmation links—the email is simply used to authenticate the session.

Once authentication completes, the SSH tunnel will open.

Accessing Your Application

After the tunnel is established, srv.us will display a public HTTPS URL such as:

```bash
https:xxxxxx.srv.us
```

Copy this link and open it in your browser (on any device).
This URL forwards directly to your local server running on 127.0.0.1:5000, allowing your app to be accessed from the internet.

Your program should now be fully accessible and running.
