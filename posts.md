---
layout: default
title: Posts
permalink: /posts/
---

<h2>Blog Posts</h2>

<ul>
  {% for post in site.posts %}
    <li>
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      <p><small>{{ post.date | date: "%B %d, %Y" }}</small></p>
      {% if post.excerpt %}
        <p>{{ post.excerpt | strip_html | truncatewords: 25 }}</p>
      {% endif %}
    </li>
  {% endfor %}
</ul>
