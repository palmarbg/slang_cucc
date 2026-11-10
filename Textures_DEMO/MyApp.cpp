#include "MyApp.h"
#include "SDL_GLDebugMessageCallback.h"

#include <imgui.h>

#include <glm/gtx/integer.hpp>

CMyApp::CMyApp() : m_cameraManipulator(m_camera)
{
}

CMyApp::~CMyApp()
{
}

void CMyApp::SetupDebugCallback()
{
	// engedélyezzük és állítsuk be a debug callback függvényt ha debug context-ben vagyunk
	GLint context_flags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
	if (context_flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
		glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DONT_CARE, 0, nullptr, GL_FALSE);
		glDebugMessageCallback(SDL_GLDebugMessageCallback, nullptr);
	}
}

void CMyApp::InitShaders()
{
	m_programID = glCreateProgram();
	AttachShader( m_programID, GL_VERTEX_SHADER, "Shaders/Vert_PosTex.vert" );
	AttachShader( m_programID, GL_FRAGMENT_SHADER, "Shaders/circle_texture_const.frag" );
	LinkProgram( m_programID );
}

void CMyApp::CleanShaders()
{
	glDeleteProgram( m_programID );
}

void CMyApp::InitGeometry()
{
	const std::initializer_list<VertexAttributeDescriptor> vertexAttribList =
	{
		{ 0, offsetof( VertexPosTex, position ), 3, GL_FLOAT },
		{ 1, offsetof( VertexPosTex, texcoord ), 2, GL_FLOAT },
	};


	// quad

	MeshObject<VertexPosTex> quadMeshCPU;

	float extend_factor = ( m_extend ? 1.0f : 0.0f );

	quadMeshCPU.vertexArray = 
	{
		{ glm::vec3( -1, -1, 0), glm::vec2( 0.0f, 0.0f ) + extend_factor * glm::vec2(-0.25, -0.25) },
		{ glm::vec3(  1, -1, 0), glm::vec2( 1.0f, 0.0f ) + extend_factor * glm::vec2( 0.25, -0.25) },
		{ glm::vec3( -1,  1, 0), glm::vec2( 0.0f, 1.0f ) + extend_factor * glm::vec2(-0.25,  0.25) },
		{ glm::vec3(  1,  1, 0), glm::vec2( 1.0f, 1.0f ) + extend_factor * glm::vec2( 0.25,  0.25) }
	};

	quadMeshCPU.indexArray =
	{
		0, 1, 2,
		1, 3, 2
	};

	m_quadGPU = CreateGLObjectFromMesh( quadMeshCPU, vertexAttribList );

}

void CMyApp::CleanGeometry()
{
	CleanOGLObject( m_quadGPU );
}

void CMyApp::InitTextures()
{
	glCreateSamplers( 1, &m_SamplerID );
	glSamplerParameteri( m_SamplerID, GL_TEXTURE_WRAP_S, m_wrapS );
	glSamplerParameteri( m_SamplerID, GL_TEXTURE_WRAP_T, m_wrapT );
	glSamplerParameteri( m_SamplerID, GL_TEXTURE_MIN_FILTER, (m_generateMipMap ?
															    ( m_linear ? GL_LINEAR_MIPMAP_LINEAR: GL_NEAREST_MIPMAP_NEAREST) :
															    ( m_linear ? GL_LINEAR : GL_NEAREST	)
	) );

	glSamplerParameteri( m_SamplerID, GL_TEXTURE_MAG_FILTER, ( m_linear ? GL_LINEAR : GL_NEAREST ) );


	ImageRGBA Image = ImageFromFile( "Assets/abstract.png" );
	unsigned int nMips = m_generateMipMap ? NumberOfMIPLevels( Image ) : 1;

	glCreateTextures( GL_TEXTURE_2D, 1, &m_TextureID );
	glTextureStorage2D( m_TextureID, nMips, GL_RGBA8, Image.width, Image.height );
	
	ImageRGBA ImageFiltered;
	ImageRGBA ImageL;

	for ( unsigned int Level = 0; Level < nMips; Level++ )
	{
		static constexpr glm::u8vec4 PIXELFILTER[ 3 ] = { {0,1,1,0}, {1,0,1,0}, {1,1,0,0} };
        ImageFiltered.Allocate( Image.width, Image.height );
        for ( unsigned int j = 0; j < Image.height; ++j )
			for ( unsigned int i = 0; i < Image.width; ++i )
			{
				glm::u8vec4 pixel = Image.GetTexel( i, j );
				if ( m_colorizeMipMaps )
				{
                       glm::u8vec4 filter = PIXELFILTER[ Level % 3 ];
					pixel = glm::u8vec4( pixel.r >> filter.r,
										 pixel.g >> filter.g,
										 pixel.b >> filter.b,
										 pixel.a >> filter.a );
				}
                   ImageFiltered.SetTexel( i, j, pixel );
               }

        glTextureSubImage2D( m_TextureID, Level, 0, 0, ImageFiltered.width, ImageFiltered.height, GL_RGBA, GL_UNSIGNED_BYTE, ImageFiltered.data() );

        ImageL.Allocate( Image.width / 2, Image.height / 2 );
		for ( unsigned int j = 0; j < ImageL.height; ++j )
			for ( unsigned int i = 0; i < ImageL.width; ++i )
			{
                   glm::u16vec4 pixel = Image.GetTexel( 2 * i, 2 * j );
                   pixel += Image.GetTexel( 2 * i + 1, 2 * j );
                   pixel += Image.GetTexel( 2 * i, 2 * j + 1 );
                   pixel += Image.GetTexel( 2 * i + 1, 2 * j + 1 );
                   pixel /= 4;

                   ImageL.SetTexel( i, j, pixel );
               }

           Image = std::move(ImageL);
	}
}

void CMyApp::CleanTextures()
{
	glDeleteTextures( 1, &m_TextureID );
    glDeleteSamplers( 1, &m_SamplerID );
}

bool CMyApp::Init()
{
	SetupDebugCallback();

	// törlési szín legyen kékes
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

	InitShaders();
	InitGeometry();
	InitTextures();

	//
	// egyéb inicializálás
	//

	glEnable(GL_CULL_FACE); // kapcsoljuk be a hátrafelé néző lapok eldobását 
	glCullFace(GL_BACK);    // GL_BACK: a kamerától "elfelé" néző lapok, GL_FRONT: a kamera felé néző lapok 

	glEnable(GL_DEPTH_TEST); // mélységi teszt bekapcsolása (takarás) 

	// kamera 
	m_camera.SetView(
		glm::vec3(0.0, 0.0, 1.0 / tanf( m_camera.GetAngle() * 0.5f ) ), // honnan nézzük a színteret  - eye
		glm::vec3(0.0, 0.0, 0.0),   // a színtér melyik pontját nézzük  - at
		glm::vec3(0.0, 1.0, 0.0));  // felfelé mutató irány a világban  - up

	m_cameraManipulator.SetStateFromCamera();

	return true;
}

void CMyApp::Clean()
{
	CleanShaders();
	CleanGeometry();
	CleanTextures();
}

void CMyApp::Update( const SUpdateInfo& updateInfo )
{
	if ( m_textureDirty )
	{
		CleanTextures();
		InitTextures();
	}

    if ( m_geometryDirty )
    {
        CleanGeometry();
        InitGeometry();
    }

	m_ElapsedTimeInSec = updateInfo.ElapsedTimeInSec;
	m_cameraManipulator.Update( updateInfo.DeltaTimeInSec );

	m_pyramidWorldTransform=glm::identity<glm::mat4>();
}

void CMyApp::Render()
{
	// töröljük a framepuffert (GL_COLOR_BUFFER_BIT)...
	// ... és a mélységi Z puffert (GL_DEPTH_BUFFER_BIT)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	// quad

	// shader bekapcsolása 
	glUseProgram(m_programID);

	// - Uniform paraméterek
	// view és projekciós mátrix
	glUniformMatrix4fv( ul("viewProj"), 1, GL_FALSE, glm::value_ptr( m_camera.GetViewProj() ) );

	glUniformMatrix4fv( ul("world"),    1, GL_FALSE, glm::value_ptr( m_pyramidWorldTransform) );

	glUniform1i( ul("textureImage"), 0 );

	// - VAO beállítása 
	glBindVertexArray( m_quadGPU.vaoID );

	// - Textúrák beállítása, minden egységre külön 
	glBindTextureUnit( 0, m_TextureID );
	glBindSampler( 0, m_SamplerID );

	// Rajzolási parancs kiadása 
	glDrawElements( GL_TRIANGLES,    
					m_quadGPU.count,			 
					GL_UNSIGNED_INT,
					nullptr );

	// shader kikapcsolása 
	glUseProgram( 0 );

	// - Textúrák kikapcsolása, minden egységre külön 
	glBindTextureUnit( 0, 0 );
	glBindSampler( 0, 0 );


	// VAO kikapcsolása 
	glBindVertexArray( 0 );
}

void CMyApp::RenderGUI()
{
	//ImGui::ShowDemoWindow();
	ImGui::Begin( "Texture DEMO" );
	m_textureDirty = ImGui::Checkbox( "Linear", &m_linear );
	m_textureDirty |= ImGui::Checkbox( "MipMap", &m_generateMipMap );

	ImGui::BeginDisabled( !m_generateMipMap );
	m_textureDirty |= ImGui::Checkbox( "Colorize MipMaps", &m_colorizeMipMaps );
	ImGui::EndDisabled();

    m_geometryDirty = ImGui::Checkbox( "Extend Textcoords", &m_extend );

    ImGui::BeginDisabled( !m_extend );
	static const struct
	{
		const char* label;
		GLint param;
	} wrapParams[ 3 ] = 
	{ 
		{"GL_REPEAT",GL_REPEAT},
		{"GL_CLAMP_TO_EDGE",GL_CLAMP_TO_EDGE},
		{"GL_MIRRORED_REPEAT",GL_MIRRORED_REPEAT}
	};
	static int wrapSIdx = 0;
	if ( ImGui::SliderInt( "Wrap S", &wrapSIdx, 0, 2, wrapParams[ wrapSIdx ].label ) )
	{
		m_wrapS = wrapParams[ wrapSIdx ].param;
		m_textureDirty = true;
	}
	static int wrapTIdx = 0;
	if ( ImGui::SliderInt( "Wrap T", &wrapTIdx, 0, 2, wrapParams[ wrapTIdx ].label ) )
	{
		m_wrapT = wrapParams[ wrapTIdx ].param;
		m_textureDirty = true;
	}
	ImGui::EndDisabled();
	ImGui::End();
}

// https://wiki.libsdl.org/SDL3/SDL_KeyboardEvent
// https://wiki.libsdl.org/SDL3/SDL_Keysym
// https://wiki.libsdl.org/SDL3/SDL_Keycode
// https://wiki.libsdl.org/SDL3/SDL_Keymod

void CMyApp::KeyboardDown(const SDL_KeyboardEvent& key)
{	
	if ( !key.repeat ) // Először lett megnyomva 
	{
		if ( key.key == SDLK_F5 && key.mod & SDL_KMOD_CTRL )
		{
			CleanShaders();
			InitShaders();
		}
		if ( key.key == SDLK_F1 )
		{
			GLint polygonModeFrontAndBack[ 2 ] = {};
			// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glGet.xhtml
			glGetIntegerv( GL_POLYGON_MODE, polygonModeFrontAndBack ); // Kérdezzük le a jelenlegi polygon módot! Külön adja a front és back módokat. 
			GLenum polygonMode = ( polygonModeFrontAndBack[ 0 ] != GL_FILL ? GL_FILL : GL_LINE ); // Váltogassuk FILL és LINE között! 
			// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glPolygonMode.xhtml
			glPolygonMode( GL_FRONT_AND_BACK, polygonMode ); // Állítsuk be az újat! 
		}
	}
	m_cameraManipulator.KeyboardDown( key );
}

void CMyApp::KeyboardUp(const SDL_KeyboardEvent& key)
{
	m_cameraManipulator.KeyboardUp( key );
}

// https://wiki.libsdl.org/SDL3/SDL_MouseMotionEvent

void CMyApp::MouseMove(const SDL_MouseMotionEvent& mouse)
{
	m_cameraManipulator.MouseMove( mouse );
}

// https://wiki.libsdl.org/SDL3/SDL_MouseButtonEvent

void CMyApp::MouseDown(const SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseUp(const SDL_MouseButtonEvent& mouse)
{
}

// https://wiki.libsdl.org/SDL3/SDL_MouseWheelEvent

void CMyApp::MouseWheel(const SDL_MouseWheelEvent& wheel)
{
	m_cameraManipulator.MouseWheel( wheel );
}


// a két paraméterben az új ablakméret szélessége (_w) és magassága (_h) található
void CMyApp::Resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h);
	m_camera.SetAspect( static_cast<float>(_w) / _h );
}

// Le nem kezelt, egzotikus esemény kezelése
// https://wiki.libsdl.org/SDL3/SDL_Event

void CMyApp::OtherEvent( const SDL_Event& ev )
{

}