Customize Ansa Script
=======================
.. _customize_ansa_script:

You can modify the original ansa data extraction script in :py:meth:`mesh2vec.mesh2vec_cae.Mesh2VecCae.from_ansa_shell` and
:py:meth:`mesh2vec.mesh2vec_cae.Mesh2VecCae.add_features_from_ansa` to include more features.
Make sure you provide also all required fields for nodes and elements (see original script below).

Original Ansa Script as Template:
----------------------------------
.. literalinclude:: ../../mesh2vec/templates/ansa.py
   :encoding: latin-1