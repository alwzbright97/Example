Jupyter Notebook Configuration(설정)
======================
-----------------------------------------------

## Change Font (폰트 변경하기)
------------------------------------------------
```
$ jupyter notebook --generate-config
$ cd ~/.jupyter/
$ mkdir custom
$ cd custom
$ vim custom.css
```

```
.CodeMirror, div.CodeMirror-code, div.output_area pre, div.output_wraaper pre {
        font-family: Consolas;
        font-size: 14px;
}

div#notebook, div.prompt {
        font-family: Consolas;
        font-size: 14px
}

.rendered_html pre, .rendered_html code {
        font-family: Consolas;
```

[Setting Code](https://gist.github.com/pmlandwehr/6bd26d0aabab5963a34dcaba1d6a18d4)


```python

```
