<!DOCTYPE html> 
<html lang="{{lang}}">
  <!-- {{dont_make_change}} -->
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <link rel="shortcut icon" href="favicon.ico">
    <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1, user-scalable=yes">
    <meta name="viewport" content="initial-scale=1.0,width=device-width">
    <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/jquery-cookie/1.4.1/jquery.cookie.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/URI.js/1.18.1/URI.min.js"></script>
    <script src="ui.js"></script>
    <title>waifu2x</title>
    <link href="style.css" rel="stylesheet" type="text/css">
    <link href="mobile.css" rel="stylesheet" type="text/css" media="screen and (max-width: 768px) and (min-width: 0px)">
    <style type="text/css">
      .option-box { margin: 4px; }
      .option-left { width: 120px; }
      .main-title { text-align: center; display: block; font-size: 1em; margin: 4px auto; }
      body { background: #ccc; }
    </style>
  </head>
  <body>
    <div class="all-page">
      <div class="main-title">waifu2x</div>
      <form action="/api" method="POST" enctype="multipart/form-data" target="_blank">
        <input type="hidden" name="recap" id="recap_response">
	<div class="option-box first">
	  <div class="option-left">{{image_choosing}}:</div>
	  <div class="option-right">
	    <input type="text" id="url" name="url" placeholder="{{type_url}}">
	    <div class="option-right-small">
	      {{choose_file}}: 
	      <input type="file" id="file" name="file"></div>
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{style}}:
	  </div>
	  <div class="option-right">
	    <label><input type="radio" name="style" class="radio" value="art" checked>
	      <span class="r-text">
		{{artwork}}
	      </span>
	    </label>
	    <label><input type="radio" name="style" class="radio" value="art_scan">
	      <span class="r-text">
		{{artwork}}/{{scan}}
	      </span>
	    </label>
	    <label><input type="radio" name="style" class="radio" value="photo">
	      <span class="r-text">
		{{photo}}
	      </span>
	    </label>
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{noise_reduction}}:
	  </div>
	  <div class="option-right">
	    <label><input type="radio" name="noise" class="radio" value="-1">
	      <span class="r-text">
		{{nr_none}}
	      </span>
	    </label>
	    <label><input type="radio" name="noise" class="radio" value="0" checked>
	      <span class="r-text">
		{{nr_low}}
	      </span>
	    </label>
	    <label><input type="radio" name="noise" class="radio" value="1" checked>
	      <span class="r-text">
		{{nr_medium}}
	      </span>
	    </label>
	    <label>
	      <input type="radio" name="noise" class="radio" value="2">
	      <span class="r-text">
		{{nr_high}}
	      </span>
	    </label>
	    <label>
	      <input type="radio" name="noise" class="radio" value="3">
	      <span class="r-text">
		{{nr_highest}}
	      </span>
	    </label>
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{upscaling}}:
	    <div class="option-left-small"></div>
	  </div>
	  <div class="option-right">
	    <label><input type="radio" name="scale" class="radio" value="-1" checked>
	      <span class="r-text">
		{{up_none}}
	      </span>
	    </label>
	    <label><input type="radio" name="scale" class="radio" value="1">
	      <span class="r-text">
		1.6x
	      </span>
	    </label>
	    <label><input type="radio" name="scale" class="radio" value="2">
	      <span class="r-text">
		2x
	      </span>
	    </label>
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{image_format}}:
	    <div class="option-left-small"></div>
	  </div>
	  <div class="option-right">
	    <label><input type="radio" name="format" class="radio" value="0" checked>
	      <span class="r-text">
                PNG
	      </span>
	    </label>
	    <label><input type="radio" name="format" class="radio" value="1">
	      <span class="r-text">
		WebP
	      </span>
	    </label>
	  </div>
	</div>
	<div id="recap_container">
	</div>
	% if button_convert:
	  <input id="submit-button" type="submit" class="button" value="{{button_convert}}">
	% else:
	  <input id="submit-button" type="submit" class="button">
	% end
	<input id="download-button" type="submit" name="download" value="{{button_download}}" class="button">
      </form>
    </div>
  </body>
</html>
