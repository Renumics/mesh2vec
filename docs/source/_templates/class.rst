:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:

    {% block methods %}

    {% if methods and  objname!="Category" %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in methods %}
      ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
      ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}





.. minigallery:: {{ module }}.{{objname}}
    :add-heading:

.. raw:: html

    <div class="clearer"></div>
