import React from 'react'
import { SidebarTrigger } from '@/components/ui/sidebar'
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuList, NavigationMenuTrigger } from '@/components/ui/navigation-menu'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Bell, HelpCircle, Search } from 'lucide-react'
import { ThemeSwitcher } from './ThemeSwitcher'

export function NavigationHeader({ user, onLogout, onGoToHome }) {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <SidebarTrigger />
          
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuTrigger>
                  Brain Tumor Detection
                </NavigationMenuTrigger>
                <NavigationMenuContent>
                  <div className="grid w-80 gap-3 p-4">
                    <div className="grid gap-2">
                      <h4 className="font-medium">MRI & Histopathology</h4>
                      <p className="text-sm text-muted-foreground">
                        EfficientNetB3-powered brain tumor detection from MRI and tissue slides
                      </p>
                    </div>
                    <div className="grid gap-2">
                      <h4 className="font-medium">Heatmap Visualization</h4>
                      <p className="text-sm text-muted-foreground">
                        AI-generated attention heatmaps highlighting suspicious regions
                      </p>
                    </div>
                  </div>
                </NavigationMenuContent>
              </NavigationMenuItem>

              <NavigationMenuItem>
                <NavigationMenuTrigger>
                  Pneumonia Detection
                </NavigationMenuTrigger>
                <NavigationMenuContent>
                  <div className="grid w-80 gap-3 p-4">
                    <div className="grid gap-2">
                      <h4 className="font-medium">Chest X-ray Analysis</h4>
                      <p className="text-sm text-muted-foreground">
                        DenseNet121 + EfficientNet-B0 ensemble model for pneumonia classification
                      </p>
                    </div>
                    <div className="grid gap-2">
                      <h4 className="font-medium">Classification Results</h4>
                      <p className="text-sm text-muted-foreground">
                        Normal vs Pneumonia prediction with confidence scores
                      </p>
                    </div>
                  </div>
                </NavigationMenuContent>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="hidden md:flex">
            {user?.role ? `${user.role} Mode` : 'System Status: Online'}
          </Badge>
          
          <ThemeSwitcher />
          
          <Button variant="ghost" size="sm">
            <Search className="h-4 w-4" />
          </Button>
          
          <Button variant="ghost" size="sm">
            <Bell className="h-4 w-4" />
            <Badge className="ml-1 h-5 w-5 rounded-full p-0 text-xs">3</Badge>
          </Button>
          
          <Button variant="ghost" size="sm">
            <HelpCircle className="h-4 w-4" />
          </Button>

          {onGoToHome && (
            <Button variant="outline" size="sm" onClick={onGoToHome}>
              Home
            </Button>
          )}
        </div>
      </div>
    </header>
  )
}