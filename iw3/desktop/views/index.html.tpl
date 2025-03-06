<!DOCTYPE html>
<html>
<head>
  <title>iw3 desktop streaming</title>
  <script>
    const FPS = ${fps};
    const WIDTH = ${frame_width};
    const HEIGHT = ${frame_height};
    const STREAM_URI = "${stream_uri}";

    window.onload = () => {
        let canvasStream = null;
        let canvas = document.createElement("canvas");
        let process_token = null;
        let stop_update = false;

        canvas.width = WIDTH;
        canvas.height = HEIGHT;

        function setup_interval(){
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.src = STREAM_URI;
            img.onload = () => {
                function render() {
                    ctx.drawImage(img, 0, 0);
                    requestAnimationFrame(render);
                }
                render();
            };
            // check server restart
            setInterval(() => {
	        if (document.hidden) {
	            return;
	        }
	        fetch('/process_token',
	              {
                          method: 'GET',
	              })
	            .then((res) => {
                        if (!res.ok) {
	                    stop_update = true;
	                    //console.log("err1", res.status, res.statusText);
                        }
                        return res.json();
	            })
	            .then((res) => {
	                //console.log("check token", process_token, res.token);
	                if (process_token == null) {
	                    process_token = res.token;
	                } else if (process_token != res.token) {
	                    // reload
	                    process_token = null;
	                    location.reload();
	                }
	            })
	            .catch((reason) => {
	                stop_update = true;
                        //console.log("err2", reason);
	            });
            }, 4000);
        }
        
        function setup_video() {
            const video = document.getElementById("player-canvas");
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, WIDTH, HEIGHT);
            canvasStream = canvas.captureStream(FPS);
            video.srcObject = canvasStream;
        }
        setup_interval();
        setup_video();
    };
  </script>
  <style type="text/css">
  body {
    margin: 0;
  }
  .video-container {
    position: fixed;
    left: 0;
    top: 0;
    z-index: 0;
    width: 100vw;
    height: 100vh;
    text-align: center;
    background-color: rgb(45, 48, 53);
  }
  .video {
    height: 100%;
    width: 100%;
    object-fit: contain;
    object-position: center;
  }
  </style>
</head>
<body>
  <div class="video-container">
    <video id="player-canvas" class="video" controls controlsList="nodownload"
	    autoplay loop muted poster="" disablepictureinpicture >
    </video>
  </div>
</body>
</html>
