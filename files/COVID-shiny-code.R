# R Shiny script for the "COVID-19 and Income Inequality" project
# Define UI for application that draws a histogram
ui <- fluidPage(theme = shinytheme("journal"),
                
                # Application title
                titlePanel("COVID-19 infection fatality rate and income inequality"),
                
                # Sidebar with a slider input for number of bins 
                sidebarLayout(
                  sidebarPanel(
                    sliderInput("range",
                                "Fatality rate age range:",
                                min = 0,
                                max = 100,
                                value = c(40,60)),
                    radioButtons("source",
                                 "Income inequality metric:",
                                 choices = list("Median income, per capita" = 1,
                                                "Gini index" = 2),
                                 selected = 1)
                  ),
                  
                  # Show a plot of the generated distribution
                  mainPanel(
                    plotOutput("IFRPlot")
                  )
                )
)

# Data and function necessary for the app
reg <- function(agemin, agemax, data){
  regdata <- data
  n_countries <- length(unique(regdata$Country))
  regdata <- regdata[regdata$AgeMidpoint >= agemin & regdata$AgeMidpoint <= agemax, ]
  
  # returns an empty data frame if # data points = 0
  if (nrow(regdata) == 0){
    return(regdata)
  }
  
  regdata <- aggregate(IFR ~ Country + MedianIncome + GiniIndex, data = regdata, FUN = mean)
  
  # returns the data set without performing the regression if the range doesn't include all countries
  if (nrow(regdata) < length(unique(data$Country))){
    return(regdata)
  }
  
  # Median income regression
  income_regression <- betareg(IFR~MedianIncome, data = regdata)
  regdata$IFR_inc = predict(income_regression, type = "response", newdata = regdata)
  regdata$pval_inc = summary(income_regression)$coef$mean["MedianIncome","Pr(>|z|)"]

  # Gini index regression
  Gini_regression <- betareg(IFR~GiniIndex, data = regdata)
  regdata$IFR_gin = predict(Gini_regression, type="response", newdata = regdata)
  regdata$pval_gin = summary(Gini_regression)$coef$mean["GiniIndex","Pr(>|z|)"]

  return(regdata)
}

# make reactive
# add regression
# add description
server <- function(input, output) {
  
  out <- reactive({
    reg(input$range[1], input$range[2], COVIDdata)
  })
  
  output$IFRPlot <- renderPlot({
    
    if (input$source == 1){
      ggplot(out(), aes(y = out()$IFR, x = out()$MedianIncome,
                        label = out()$Country))+
        geom_point()+
        
        # Regression layer
        {if (nrow(out()) == length(unique(COVIDdata$Country)) 
             & "pval_inc" %in% colnames(out()))
          
          if (out()$pval_inc[1] < 0.05){
            geom_line(data = out(), aes(x = out()$MedianIncome,y = out()$IFR_inc))
          }
        }+
        
        geom_text_repel(size = 4)+
        theme(legend.position = "none")+
        theme(axis.title = element_text(size = 25), axis.text = element_text(size = 25),
              strip.text.x = element_text(size = 15))+
        scale_y_continuous(labels = scales::percent)+
        scale_x_continuous(breaks = seq(5000, 20000, 5000), labels = c("5,000", "10,000", "15,000", "20,000"))+
        xlab("Median income ($)")+ 
        ylab("Age-specific infection fatality rate")
    } else {
      ggplot(out(), aes(y = out()$IFR, x = out()$GiniIndex,
                        label = out()$Country))+
        geom_point()+
        
        {if (nrow(out()) == length(unique(COVIDdata$Country)) 
             & "pval_inc" %in% colnames(out()))
          
          if (out()$pval_inc[1] < 0.05){
            geom_line(data = out(), aes(x = out()$GiniIndex,y = out()$IFR_gin))
          }
        }+

        
        geom_text_repel(size = 4)+
        theme(legend.position = "none")+
        theme(axis.title = element_text(size = 25), axis.text = element_text(size = 25),
              strip.text.x = element_text(size = 15))+
        scale_y_continuous(labels = scales::percent)+
        xlab("Gini Index")+ 
        ylab("Age-specific infection fatality rate")
    }
  })
}

# Run the application 
shinyApp(ui = ui, server = server, options = list(height = 500))
