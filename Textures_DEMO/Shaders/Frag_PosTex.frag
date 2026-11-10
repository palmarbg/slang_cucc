#version 430

// pipeline-ból bejövő per-fragment attribútumok 
in vec2 textureCoords;

// kimenő érték - a fragment színe 
out vec4 outputColor;

// textúra mintavételező objektum 
uniform sampler2D textureImage;

void main()
{
	outputColor = texture( textureImage, textureCoords );
}