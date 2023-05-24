<!DOCTYPE html> 
<html lang="{{lang}}">
  <!-- {{dont_make_change}} -->
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <link rel="shortcut icon" href="favicon.ico">
    <meta name="viewport" content="initial-scale=1.0,width=device-width">
    <link href="//cdnjs.cloudflare.com/ajax/libs/normalize/3.0.3/normalize.min.css" rel="stylesheet" type="text/css">
    <link href="style.css" rel="stylesheet" type="text/css">
    <link href="mobile.css" rel="stylesheet" type="text/css" media="screen and (max-width: 768px) and (min-width: 0px)">
    <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/jquery-cookie/1.4.1/jquery.cookie.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/URI.js/1.18.1/URI.min.js"></script>
    <script src="ui.js"></script>
    <title>waifu2x</title>
  </head>
  <body>
    <div class="all-page">
      <h1 class="main-title">waifu2x</h1>
      <div class="choose-lang">
	<a href="index.html">
	  English
	</a>
	/
	<a href="index.ja.html">
	  日本語
	</a>
	/
	<a href="index.ru.html">
	  Русский
	</a>
	/
	<a href="index.pt.html">
	  Português
	</a>
	/
	<a href="index.es.html">
	  Español
	</a>
	/
	<a href="index.fr.html">
	  Français
	</a>
	/
	<a href="index.de.html">
	  Deutsch
	</a>
	/
	<a href="index.tr.html">
	  Türkçe
	</a>
	/
	<a href="index.zh-CN.html">
	  简体中文
	</a>
	/
	<a href="index.zh-TW.html">
	  繁體中文
	</a>
	/
	<a href="index.ko.html">
	  한국어
	</a>
	/
	<a href="index.nl.html">
	  Nederlands
	</a>
	/
	<a href="index.ca.html">
	  Català
	</a>
	/
	<a href="index.ro.html">
	  Română
	</a>
	/
	<a href="index.it.html">
	  Italiano
	</a>
	/
	<a href="index.eo.html">
	  Esperanto
	</a>
	/
	<a href="index.no.html">
	  Bokmål
	</a>
	/
	<a href="index.uk.html">
	  Українська
	</a>
	/
	<a href="index.pl.html">
	  Polski
	</a>
	/
	<a href="index.bg.html">
	  Български
	</a>
      </div>
      <p>{{description}}</p>
      <p class="margin1 link-box">
	<a href="https://raw.githubusercontent.com/nagadomi/nunif/master/waifu2x/docs/images/slide.png" class="blue-link" target="_blank">
	  {{show_demonstration}}
	</a>
	| 
	<a href="https://github.com/nagadomi/nunif" class="blue-link" target="_blank">
	  {{go_to_github}}
	</a>
      </p>
      <form action="/api" method="POST" enctype="multipart/form-data" target="_blank">
        <input type="hidden" name="recap" id="recap_response">
	<div class="option-box first">
	  <div class="option-left">{{image_choosing}} (D&amp;D):</div>
	  <div class="option-right">
	    <input type="text" id="url" name="url" placeholder="{{type_url}}">
	    <div class="option-right-small">
	      {{choose_file}}: 
	      <input type="file" id="file" name="file"></div>
	  </div>
	  <div class="option-hint file_limits">
	    {{file_limits}}
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{style}}:
	  </div>
	  <div class="option-right">
	    <label><input type="radio" name="style" class="radio" value="art" checked>
	      <span class="r-text" title="Anime Style Art, Cliparts">
		{{artwork}}
	      </span>
	    </label>
	    <label><input type="radio" name="style" class="radio" value="art_scan">
	      <span class="r-text" title="Manga, Anime Screencaps, Anime Style Art for more clear results">
		{{artwork}}/{{scan}}
	      </span>
	    </label>
	    <label><input type="radio" name="style" class="radio" value="photo">
	      <span class="r-text" title="Photograph">
		{{photo}}
	      </span>
	    </label>
	  </div>
	</div>
	<div class="option-box">
	  <div class="option-left">
	    {{noise_reduction}}:
	    <div class="option-left-small">
	      ({{expect_jpeg}})
	    </div>
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
	  <div class="option-hint">
	    {{nr_hint}}
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
	<div class="bottom-hint">
	  <ul>
	    % for hint in hints:
	      <li>{{hint}}</li>
	    % end
	  </ul>
	</div>
      </form>
    </div>
    <div class="bottom-info address">
      <a href="https://github.com/nagadomi/waifu2x" class="gray-link" target="_blank">waifu2x</a>,
      <a href="https://github.com/nagadomi/nunif" class="gray-link" target="_blank">nunif</a>
    </div>
  </body>
</html>
