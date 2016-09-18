
function phase = Compute_vPhase(DataSource,itnum,phase_range,Xres,Yres)
  %lon=get_lon(DataSource);    % provide longitude vector
  %lat=get_lat(DataSource);    % provide latitude vector

  [LON,LAT]=meshgrid(lon,lat);
  min_lon=min(lon);
  max_lon=max(lon);
  min_lat=min(lat);
  max_lat=max(lat);

  if Xres & Yres
    X=(min_lon):Xres:(max_lon);
    Y=(min_lat):Yres:(max_lat);
  else
    X=lon;
    Y=lat;
  end
  [X,Y]=meshgrid(X,Y);

  %V=getV_obs(DataSource,itnum);    % provide U matrix
  %U=getU_obs(DataSource,itnum);    % provide V matrix

  U=interpolate(LON,LAT,U',X,Y);
  V=interpolate(LON,LAT,V',X,Y);
  phase=vPhase(U,V,phase_range);

  %plotPhase(phase,{'vPhase',num2str(itnum)});
 
  if phase_range==2*pi
    save(sprintf('provide file path',itnum),'phase','itnum','phase_range','Xres','Yres')	
  end

  if phase_range==pi
    save(sprintf('provide file path',itnum),'phase','itnum','phase_range','Xres','Yres')
  end

  if phase_range==pi/2
    save(sprintf('provide file path',itnum),'phase','itnum','phase_range','Xres','Yres')
  end
  
end


function plotPhase(phase,ti)
  figure;
  colormap jet;
  imagesc(phase);
  axis xy;
  colorbar;
  title(ti);
  daspect([1 1 1]);
  drawnow; 
  grid on;
end
